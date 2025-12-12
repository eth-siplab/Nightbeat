import polars as pl
import numpy as np
import os
from helpers.data_loader import load_test_data_nightbeatdb, load_test_data_oliviawalch, get_base_paths

class BCGProcessor:
    def __init__(self, algo_name, process_window_func, output_schema, fs=100):
        self.algo_name = algo_name
        self.process_window_func = process_window_func
        self.output_schema = output_schema
        self.fs = fs

    def load_data(self, subject, all_subjects, dataset, rows):
        if dataset == 'nightbeatdb':
            return load_test_data_nightbeatdb(subject_id=all_subjects[subject], windows=rows)
        elif dataset == 'aw':
            return load_test_data_oliviawalch(subject_id=all_subjects[subject], windows=rows)
        else:
            raise ValueError('Dataset not recognized')

    def process_df(self, df, **kwargs):
        # Accumulators
        data_acc = {key: [] for key in self.output_schema.keys()}
        
        # Pre-calculation hook (if needed by specific algo wrappers, they handles it in kwargs)
        
        for i, row in enumerate(df.iter_rows(named=True)):
            # if i % 100 == 0:
            #     print(f"Processing row {i} out of {len(df)}")

            # Extract time early
            t_val = row.get('start_100Hz') or row.get('start')
            
            # Call specific algorithm logic
            # output must be a dict matching the schema keys (excluding row_idx/time if handled here)
            try:
                result = self.process_window_func(row, self.fs, **kwargs)
                
                # Append Results
                data_acc['row_idx'].append(i)
                data_acc['time'].append(t_val)
                for k, v in result.items():
                    data_acc[k].append(v)
            except Exception as e:
                # Basic error handling to skip bad windows
                print(f"Error processing row {i}: {e}")
                continue

        # Post-processing: Filter outliers based on spectrogram shape (stft outputs)
        # Assuming 'magnitude' is the key for STFT
        if 'magnitude' in data_acc and len(data_acc['magnitude']) > 0:
            heights = [np.shape(m)[0] for m in data_acc['magnitude']]
            widths = [np.shape(m)[1] for m in data_acc['magnitude']]

            # check that all heights and widths ore as required: widhth in (946, 951), height = 359
            # assert all(h == 359 for h in heights), "Unexpected height in STFT magnitude"
            # assert all(w == 951 for w in widths), "Unexpected width in STFT magnitude"

            # Filter indices
            # NOTE: Adjust these dimensions if you change STFT settings in process_window_func
            # NOTE: On the aw dataset, some windows have width 946 instead of 951 due to slight differences in windowing, we simply reduce to 946 in all cases
            keep_indices = [
                i for i in range(len(heights)) 
                if heights[i] == 359 and widths[i] in (946, 951)
            ]

            if len(keep_indices) < len(heights):
                print(f"There were {len(heights) - len(keep_indices)} windows where the STFT shape did not match (359, 946) or (359, 951). These windows will be skipped.")
                print(f"Most common heights: {np.unique(heights, return_counts=True)}")
                print(f"Most common widths: {np.unique(widths, return_counts=True)}")

            # Filter data_acc accordingly
            for k in data_acc.keys():
                data_acc[k] = [data_acc[k][i] for i in keep_indices]

            data_acc['magnitude'] = [m[:, :946] if np.shape(m)[1] == 951 else m for m in data_acc['magnitude']]

        # Construct DataFrame
        # Convert numpy arrays to lists for Polars compatibility where necessary
        for k in data_acc:
            data_acc[k] = [v.tolist() if isinstance(v, np.ndarray) else v for v in data_acc[k]]

        df_out = pl.DataFrame(data_acc, schema=self.output_schema)
        return df_out

    def load_and_process(self, subject, all_subjects, dataset, rows, **kwargs):
        df = self.load_data(subject, all_subjects, dataset, rows)
        print(f"Processing subject {all_subjects[subject]} in dataset {dataset} with {min(rows, len(df))} rows")

        df_processed = self.process_df(df, **kwargs)

        # Save Logic
        base_path = get_base_paths(transformed=True)[0 if dataset == 'nightbeatdb' else 1]
        out_name = f'{self.algo_name}_{dataset}_{int(all_subjects[subject]):02d}.parquet'
        out_path = os.path.join(base_path, out_name)
        
        print(f"Saving in {out_path}")
        df_processed.write_parquet(out_path, compression='zstd')
        
        return 'saved'