from pathlib import Path


class Vocab:
    def __init__(self, directory):
        """
        Initialize the Vocab class with the directory and validate vocab files.

        Parameters:
        -----------
        directory : str or Path
            The path to the directory where the files are located.
        """
        self.directory = Path(directory)
        self.valid_files = self._get_and_check_files()

    def _get_and_check_files(self):
        """
        Private method to find and validate vocab files in the directory.

        Returns:
        --------
        list
            A list of file paths that are valid vocab files.
        """
        if not self.directory.is_dir():
            raise ValueError(
                f"The specified directory {self.directory} does not exist."
            )

        valid_files = []
        for file_path in self.directory.glob("*vocab.txt"):
            if file_path.suffix == ".txt":
                with file_path.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if all(line.strip() for line in lines):  # Ensure no empty lines
                        valid_files.append(file_path)
                    else:
                        print(f"File {file_path} contains empty lines.")
        return valid_files

    def get_valid_files(self):
        """
        Public method to return the list of valid vocab files.

        Returns:
        --------
        list
            A list of valid vocab file paths.
        """
        return self.valid_files

    def generate_combined_vocab(self):
        """
        Generate a string that concatenates all lines from all valid vocab files.

        Returns:
        --------
        str
            A single string containing all lines concatenated from all vocab files.
        """
        combined_vocab = []

        for file_path in self.valid_files:
            with file_path.open("r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                combined_vocab.extend(lines)  # Add each line to the combined vocab list

        return "\n".join(
            combined_vocab
        )  # Return all lines joined by a newline character

    def export_combined_vocab_to_txt(self, output_dir, name="combined_vocab"):
        """
        Export the combined vocab string to a .txt file.

        Parameters:
        -----------
        output_file : str or Path
            The path to the output .txt file where the combined vocab will be saved.

        Returns:
        --------
        None
        """
        # Get the combined vocab string
        combined_vocab = self.generate_combined_vocab()

        # Write the combined vocab to the specified output file
        output_path = Path(output_dir) / Path(name + "_vocab.txt")
        with output_path.open("w", encoding="utf-8") as f:
            f.write(combined_vocab)

        print(f"Combined vocab saved to {output_path}")
