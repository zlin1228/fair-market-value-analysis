import os
import pandas as pd
import sys

def main():
    if len(sys.argv) < 2:
        print(
            'Usage: <history_file_prefix>\nThe history file path prefix is the prefix of the \n" + '
            '"history files to be merged.\n' +
            'The history files are expected to be in the same directory as the history file prefix.\n' +
            'The history files are expected to be named <history_file_prefix><batch size>\n' +
            'The history files are consolidated in order by batch size.\n' +
            'The consolidated history file is named <history_file_prefix>.consolidated.csv\n'
        )
        exit()

    history_file_prefix = sys.argv[1]

    # Loop through all the history files and consolidate them.
    # First list the history files which start with the prefix.
    history_prefix_directory_path = os.path.dirname(history_file_prefix)
    history_prefix_file_name = os.path.basename(history_file_prefix)
    # Get files in the directory path which start with the prefix.
    history_files = [f for f in os.listdir(history_prefix_directory_path) if (f.startswith(history_prefix_file_name)
                                                                              and not f.endswith('.consolidated.csv'))]
    # Sort the history files by batch size.
    history_files.sort(key=lambda f: int(f[len(history_prefix_file_name):]))
    # Consolidate the history files.
    consolidated_history = []
    for history_file in history_files:
        history_file_path = os.path.join(history_prefix_directory_path, history_file)
        history = pd.read_csv(history_file_path)
        batch_size = int(history_file[len(history_prefix_file_name):])
        history.insert(0, 'batch_size', batch_size)
        consolidated_history.append(history)

    consolidated_history = pd.concat(consolidated_history, ignore_index=True)

    # Change the name of the epoch column to batch_size_epoch.
    consolidated_history.rename(columns={'epoch': 'batch_size_epoch'}, inplace=True)

    # Create a new epoch column which is the epoch number for the entire training, which is 0 indexed
    consolidated_history.insert(0, 'epoch', range(0, len(consolidated_history)))

    # Write the consolidated history to a file.
    consolidated_history_file_path = history_file_prefix + '.consolidated.csv'
    consolidated_history.to_csv(consolidated_history_file_path, index=False)


if __name__ == "__main__":
    main()
