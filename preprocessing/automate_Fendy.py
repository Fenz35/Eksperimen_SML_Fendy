import pandas as pd
import os

from preprocessing_fendy import preprocess_data

def run_automation():
    # Atur path penyimpanan
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_path = os.path.join(script_dir, '..', 'Gaming_Academic_Performance.csv')
    output_dir = os.path.join(script_dir, 'GAP_preprocessing')

    save_model_path = os.path.join(output_dir, 'preprocessor.joblib')
    header_path = os.path.join(output_dir, 'data_headers.csv')
    processed_file = os.path.join(output_dir, 'train_cleaned.csv')

    os.makedirs(output_dir, exist_ok=True)

    try:
        # Memuat dataset
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file tidak ditemukan: {input_path}")
        df = pd.read_csv(input_path)
        
        # Jalankan preprocessing
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(
            data=df, 
            target_column='grades', 
            save_path=save_model_path, 
            header_path=header_path
        )
    
        # Menyimpan Hasilnya
        combined_df = pd.DataFrame(X_train, columns=feature_names)
        combined_df['grades'] = y_train.values
        combined_df.to_csv(processed_file, index=False)
        
        print(f"Data yang telah diproses disimpan ke: {processed_file}")
        print(f"Preprocessor pipeline disimpan ke: {save_model_path}")
        print(f"Nama kolom disimpan ke: {header_path}")
    
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    run_automation()