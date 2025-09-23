import os
import glob
from typing import Dict, List, Optional


class FileMap:
    """
    A class to map filenames to their relative paths within a directory.
    """
    
    def __init__(self, directory_path: str, recursive: bool = False, file_extensions: Optional[List[str]] = None):
        """
        Initialize FileMap with a directory path.
        
        Args:
            directory_path (str): Path to the directory to scan
            recursive (bool): Whether to scan subdirectories recursively
            file_extensions (List[str], optional): List of file extensions to include (e.g., ['.wav', '.mp3']). 
                                                 If None, includes all files.
        """
        self.directory_path = directory_path
        self.recursive = recursive
        self.file_extensions = file_extensions
        self.file_map: Dict[str, str] = {}
        self._build_file_map()
    
    def _build_file_map(self):
        """Build the file mapping dictionary."""
        if not os.path.exists(self.directory_path):
            raise ValueError(f"Directory '{self.directory_path}' does not exist")
        
        if not os.path.isdir(self.directory_path):
            raise ValueError(f"'{self.directory_path}' is not a directory")
        
        # Determine the pattern for file search
        if self.recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        # Get all files matching the pattern
        search_pattern = os.path.join(self.directory_path, pattern)
        all_files = glob.glob(search_pattern, recursive=self.recursive)
        
        # Filter files and build the map
        for file_path in all_files:
            if os.path.isfile(file_path):
                # Check file extension if specified
                if self.file_extensions:
                    _, ext = os.path.splitext(file_path)
                    if ext.lower() not in [e.lower() for e in self.file_extensions]:
                        continue
                
                # Get relative path from current working directory
                rel_path = os.path.relpath(file_path)
                # Convert backslashes to forward slashes
                rel_path = rel_path.replace('\\', '/')
                # Add ./ prefix to relative path
                if not rel_path.startswith('./'):
                    rel_path = './' + rel_path
                filename = os.path.basename(file_path)
                # Remove file extension from filename
                filename = os.path.splitext(filename)[0]
                
                # Store in map
                self.file_map[filename] = rel_path
    
    def get_path(self, filename: str) -> Optional[str]:
        """
        Get the relative path for a given filename.
        
        Args:
            filename (str): The filename to search for
            
        Returns:
            Optional[str]: The relative path if found, None otherwise
        """
        return self.file_map.get(filename)
    
    def get_all_files(self) -> Dict[str, str]:
        """
        Get all file mappings.
        
        Returns:
            Dict[str, str]: Dictionary mapping filenames to relative paths
        """
        return self.file_map.copy()
    
    def get_filenames(self) -> List[str]:
        """
        Get all filenames in the map.
        
        Returns:
            List[str]: List of all filenames
        """
        return list(self.file_map.keys())
    
    def get_paths(self) -> List[str]:
        """
        Get all relative paths in the map.
        
        Returns:
            List[str]: List of all relative paths
        """
        return list(self.file_map.values())
    
    def refresh(self):
        """Refresh the file map by rescanning the directory."""
        self.file_map.clear()
        self._build_file_map()
    
    def __len__(self):
        """Return the number of files in the map."""
        return len(self.file_map)
    
    def __contains__(self, filename: str):
        """Check if a filename exists in the map."""
        return filename in self.file_map
    
    def __getitem__(self, filename: str):
        """Get path for filename using dictionary-style access."""
        return self.file_map[filename]


def get_wav_filenames(directory_path):
    """
    Read all .wav files from a directory and return only the filenames without path and extension.
    
    Args:
        directory_path (str): Path to the directory containing .wav files
        
    Returns:
        list: List of filenames without path and .wav extension
    """
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory '{directory_path}' does not exist")
    
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a directory")
    
    # Use glob to find all .wav files in the directory
    wav_pattern = os.path.join(directory_path, "*.wav")
    wav_files = glob.glob(wav_pattern)
    
    # Extract filenames without path and extension
    filenames = []
    for file_path in wav_files:
        filename = os.path.basename(file_path)  # Get filename without path
        name_without_ext = os.path.splitext(filename)[0]  # Remove .wav extension
        filenames.append(name_without_ext)
    
    return filenames


def get_wav_filenames_case_insensitive(directory_path):
    """
    Read all .wav files from a directory (case insensitive) and return only the filenames without path and extension.
    
    Args:
        directory_path (str): Path to the directory containing .wav files
        
    Returns:
        list: List of filenames without path and .wav extension
    """
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory '{directory_path}' does not exist")
    
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a directory")
    
    # Use glob to find all .wav files (case insensitive)
    wav_pattern = os.path.join(directory_path, "*.[Ww][Aa][Vv]")
    wav_files = glob.glob(wav_pattern)
    
    # Extract filenames without path and extension
    filenames = []
    for file_path in wav_files:
        filename = os.path.basename(file_path)  # Get filename without path
        name_without_ext = os.path.splitext(filename)[0]  # Remove .wav extension
        filenames.append(name_without_ext)
    
    return filenames
