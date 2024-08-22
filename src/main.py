from utils import *
import toml
import numpy as np
import pandas as pd
import cv2
import os
import argparse

### Generates spectrogram of audio event detections

# Main
def main():
    # Arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('option_file', 
                        help='Path to the option file (.toml format)')
    args = parser.parse_args()

    with open(args.option_file, 'r') as file:
        toml_data = toml.load(file)

    # .toml file variables    
    piezo_audio_paths = toml_data['paths']['piezo_audio_paths']
    label_paths = toml_data['paths']['label_paths']
    write_dir = toml_data['paths']['img_write_dir']
    edge_dir = toml_data['paths']['edge_write_dir']
    csv_write_path = toml_data['paths']['csv_write_path']


    df_list = []
    
    # Loop through audio paths
    for index, audio in enumerate(piezo_audio_paths):
        audio_name = audio.split('/')[1].split(".")[0]
        make_directories(write_dir, audio_name) # Images
        make_directories(edge_dir, audio_name) # edges

        #print(piezo_audio_paths)
        #input('')

        # Create Dataset
        fs = 30_000
        df = pd.read_csv(label_paths[index], 
                        sep='\t', 
                        header=None)
        
        df.rename(columns={0: 'onset', 1: 'offset', 2: 'label'}, inplace=True)
        df['type'] = df['label'].apply(lambda x: 'voc' if x == 1 else 'noise')
        print(df)
        print(df['label'].unique())
        df['onset_sample'] = (df['onset'] * fs).astype(int)
        df['offset_sample'] = (df['offset'] * fs).astype(int)
        df['length'] = df['offset_sample'] - df['onset_sample']
    
        df['path'] = df.apply(lambda row: os.path.join(write_dir, audio_name, row['type'], f"{audio_name}_{row['onset_sample']}.jpg").replace('\\', '/'), axis=1)
        df['edge_path'] = df.apply(lambda row: os.path.join(edge_dir, audio_name, row['type'], f"{audio_name}_{row['onset_sample']}.jpg").replace('\\', '/'), axis=1)
        df['bird'] = str(audio_name)

        category_counts = df['label'].value_counts()
        print(category_counts)
        df_list.append(df)
        print(df)

        # Load and preprocess audio file
        [y_piezo, fs] = load_filter_audio(piezo_audio_paths[index])

        # Iterate across rows
        for index, row in df.iterrows():
            start = int(row['onset_sample'])
            end = int(row['offset_sample'])

            piezo_temp = y_piezo[start:end]

            temp = np.zeros(int(fs*0.224))

            if len(piezo_temp) > len(temp):
                temp = piezo_temp[0:len(temp)]
            else:
                temp[0:len(piezo_temp)] = piezo_temp

            """# Tiling the audio
            [mul, rem] = np.divmod(fs*0.254,len(piezo_temp))
            #piezo_temp = np.tile(piezo_temp, int(mul+1))
            #piezo_temp = piezo_temp[0:int(fs*0.224)]"""

            stftMat_mel = generate_spectrogram(temp, fs) # Create Mel Spectrogram
            normalized_image1 = spec2dB(stftMat_mel) # Convert to dB and apply CLAHE
            gray = cv2.cvtColor(normalized_image1, cv2.COLOR_BGR2GRAY) # Convert to grayscale
            edges = cv2.Canny(gray, 190, 200) # Canny edge detection

            print(f"\t({index}) Writing: {row['path']}, {normalized_image1.shape}")
            cv2.imwrite(row['path'], normalized_image1) # Write image
            cv2.imwrite(row['edge_path'], edges) # Write image

    ls = []

    # Merge DFs
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Divide into fold
    merged_df['fold'] = pd.cut(merged_df.index, bins=5, labels=False)
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = merged_df[merged_df['fold'].isin([0,1,2,3])]
    train_df['fold'] = pd.cut(train_df.index, bins=5, labels=False)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df  = train_df .sort_values(by='fold').reset_index(drop=True)

    # Renumber test set fold
    test_set = merged_df[merged_df['fold'] == 4].reset_index(drop=True)
    test_set['fold'] = 5

    ls.append(train_df)
    ls.append(test_set)

    # Save merged df
    merged_df = pd.concat(ls, ignore_index=True)
    merged_df.to_csv(csv_write_path, index=False)
    print(merged_df)

if __name__ == "__main__":
    main()