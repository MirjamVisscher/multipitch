
#!/usr/bin/env python3
# environment multipitch_gpu
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 18:09:32 2024

@author: mirjam
"""
import argparse
import os
import numpy as np
import pandas as pd
import librosa
import torch
from libdl.data_preprocessing import compute_efficient_hcqt
from libdl.data_loaders import dataset_context
from libdl.nn_models import simple_u_net_polyphony_classif_softmax, deep_cnn_segm_sigmoid
from torchinfo import summary

print("torch cuda is available" + str(torch.cuda.is_available()))  # changeID 2
print(torch.cuda.device_count())  # changeID 2
if not torch.cuda.is_available(): # changeID 8
    raise RuntimeError("CUDA is not available. Check your GPU configuration.") # changeID 8


# Argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='Run multipitch detection.')
    parser.add_argument('--experiment', required=True, help='Name of the experiment directory.')
    parser.add_argument('--model', required=True, choices=['195f', '214c'], help='Model to use.')
#*    parser.add_argument('--audio_folder', required=True, help='Folder containing audio files.')
#*    parser.add_argument('--audio_file', help='Single audio file for processing.')
    parser.add_argument('--type', required=True, choices=['folder', 'file'], help='Specify if the input is a folder or a single file.')
    parser.add_argument('--path', required=True, help='Path to the audio folder or file.')
    return parser.parse_args()

# Main function
def main():
    try:
        # Parse arguments
        args = parse_arguments()
        experiment = args.experiment
        model_name = args.model
#*        audio_folder = args.audio_folder
#*        audio_file = args.audio_file
        input_type = args.type  # 'folder' or 'file'
        path = args.path  # Path to the folder or file


#*        # Validate input arguments
#*        if not audio_folder and not audio_file:
#*            raise ValueError("You must specify either --audio_folder or --audio_file.")
#*        if audio_folder and audio_file:
#*            raise ValueError("Specify either --audio_folder or --audio_file, not both.")

        # Set up paths and parameters
        basepath = os.path.abspath(os.path.dirname(__file__)) #changeID 9
#        basepath = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) # changeID 9
        dir_models = os.path.join(basepath, 'models_pretrained')
        num_octaves_inp = 6
        num_output_bins, min_pitch = 72, 24

        # Initialize model based on the provided argument
        if model_name == '195f':
            model_params = {
                'n_chan_input': 6,
                'n_chan_layers': [128, 180, 150, 100],
                'n_ch_out': 2,
                'n_bins_in': num_octaves_inp * 12 * 3,
                'n_bins_out': num_output_bins,
                'a_lrelu': 0.3,
                'p_dropout': 0.2,
                'scalefac': 2,
                'num_polyphony_steps': 24
            }
            fn_model = 'RETRAIN4_exp195f_musicnet_aligned_unet_extremelylarge_polyphony_softmax_rerun1.pt'
            model = simple_u_net_polyphony_classif_softmax(
                n_chan_input=model_params['n_chan_input'],
                n_chan_layers=model_params['n_chan_layers'],
                n_bins_in=model_params['n_bins_in'],
                n_bins_out=model_params['n_bins_out'],
                a_lrelu=model_params['a_lrelu'],
                p_dropout=model_params['p_dropout'],
                scalefac=model_params['scalefac'],
                num_polyphony_steps=model_params['num_polyphony_steps']
            )
        elif model_name == '214c':
            model_params = {'n_chan_input': 6, 
                    'n_chan_layers': [40,40,30,10], 
                    'n_prefilt_layers': 5, 
                    'residual': True, 
                    'n_bins_in': num_octaves_inp*12*3, 
                    'n_bins_out': num_output_bins, 
                    'a_lrelu': 0.3, 
                    'p_dropout': 0.2
                   }
            mp = model_params

            fn_model = 'exp214c_bigmix_aligned_cnn_deepresnetwide.pt'
            model = deep_cnn_segm_sigmoid(n_chan_input=mp['n_chan_input'], 
                                          n_chan_layers=mp['n_chan_layers'], 
                                          n_prefilt_layers=mp['n_prefilt_layers'],
                                          residual=mp['residual'], 
                                          n_bins_in=mp['n_bins_in'], 
                                          n_bins_out=mp['n_bins_out'], 
                                          a_lrelu=mp['a_lrelu'], 
                                          p_dropout=mp['p_dropout']
                                         )


        else:
            raise ValueError("Please specify an existing model. You can choose between '214c' and '195f'")

        # Load model
        path_trained_model = os.path.join(dir_models, fn_model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #changeID 7
        model = model.to(device) #changeID 7

        model.load_state_dict(torch.load(path_trained_model, map_location=device)) # set to 'cuda' if dependencies are solved
        model.eval()
        #summary(model, input_size=(1, 6, 174, 216))

        fs = 22050
        if input_type == 'file':
            audiofiles = [path]  # Directly use the file path # changeID 15
        else:  # input_type == 'folder'
            output_path = os.path.join(basepath, 'output', 'predictions', experiment, model_name) # changeID 4
            os.makedirs(output_path, exist_ok=True)

            recordings = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav')] # changeID 4
            existing_csv_files = {os.path.splitext(f)[0] for f in os.listdir(output_path) if f.endswith('.csv')} # changeID 4
            audiofiles = [f for f in recordings if os.path.splitext(os.path.basename(f))[0] not in existing_csv_files] # changeID 4

        for recording in audiofiles: #changeID 4
            try:
                f_audio, fs_load = librosa.load(recording, sr=fs)
                bins_per_semitone = 3
                num_octaves = 6
                num_harmonics = 5
                num_subharmonics = 1
                center_bins = True

                f_hcqt, fs_hcqt, hopsize_cqt = compute_efficient_hcqt(
                    f_audio,
                    fs=22050,
                    fmin=librosa.note_to_hz('C1'),
                    fs_hcqt_target=50,
                    bins_per_octave=bins_per_semitone * 12,
                    num_octaves=num_octaves,
                    num_harmonics=num_harmonics,
                    num_subharmonics=num_subharmonics,
                    center_bins=center_bins
                )
                print('loaded: '+ recording)
                # Set test parameters
                test_params = {'batch_size': 16, 'shuffle': False, 'num_workers': 4} # changeID 6 and 7
#                test_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 1}
                test_dataset_params = {'context': 75, 'stride': 1, 'compression': 10}
                half_context = test_dataset_params['context'] // 2
#                print('set test parameters 1')

                inputs = np.transpose(f_hcqt, (2, 1, 0))
                targets = np.zeros(inputs.shape[1:])  # need dummy targets to use dataset object
#                print('set test parameters ')

                inputs_context = torch.from_numpy(np.pad(inputs, ((0, 0), (half_context, half_context + 1), (0, 0))))
                targets_context = torch.from_numpy(np.pad(targets, ((half_context, half_context + 1), (0, 0))))
#                print('set test parameters 3')

                test_set = dataset_context(inputs_context, targets_context, test_dataset_params)
                test_generator = torch.utils.data.DataLoader(test_set, **test_params)
#                print('set test parameters 4')

                pred_tot = np.zeros((0, num_output_bins))
                k = 0

                for test_batch, test_labels in test_generator:
                    test_batch = test_batch.to(device) # changeID 7
                    if k == 1:
                        print('batch ' + str(k) + ' ' + recording)
                    if k % 10 == 0:
                        print('batch ' + str(k) + ' ' + recording)
                    k += 1
                    if model_name == '195f':
                        y_pred, n_pred = model(test_batch)
                    if model_name == '214c': 
                        y_pred = model(test_batch)
 #                   print('y_pred set')

                    pred_log = torch.squeeze(torch.squeeze(y_pred.to('cpu'), 2), 1).detach().numpy()
 #                   print('pred_log set')
                    pred_tot = np.append(pred_tot, pred_log, axis=0)
 #                   print('pred_tot set')

                predictions = pd.DataFrame(pred_tot)
                output_path = os.path.join(basepath, 'output', 'predictions', experiment, model_name)
                os.makedirs(output_path, exist_ok=True)
                file_name = os.path.splitext(os.path.basename(recording))[0] + '.csv'
#*                file_name = os.path.splitext(recording)[0] + '.csv'
                path_prediction = os.path.join(output_path, file_name)
                predictions.to_csv(path_prediction)

            except Exception as e:
                print(f"Error processing file {recording}: {e}")

    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == '__main__':
    main()

