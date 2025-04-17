import fsspec
from typing import BinaryIO, List
from datetime import datetime
from src.logging_config import Logger


class CloudStorage:
    def __init__(self, cloud_provider="gcp"):
        self.cloud_provider = cloud_provider.lower()
        self.logger = Logger.create_logger(__name__)

        if self.cloud_provider == "gcp":
            self.fs = fsspec.filesystem("gs")  # Google Cloud Storage
        elif self.cloud_provider == "aws":
            self.fs = fsspec.filesystem("s3")  # Amazon S3
        else:
            raise ValueError("Invalid cloud_provider.  Must be 'gcp' or 'aws'.")

    def upload_object(self, file_obj: BinaryIO, bucket_name: str, object_name: str) -> str:
        try:
            cloud_path = f"{self._get_protocol()}://{bucket_name}/{object_name}"
            with self.fs.open(cloud_path, "wb") as f:
                f.write(file_obj.read())

            if self.logger:
                self.logger.info(f"File uploaded to {cloud_path}")
            return cloud_path
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error uploading file to {self.cloud_provider.upper()}: {str(e)}")
            raise

    def download_object(self, cloud_path: str) -> bytes:
        try:
            if not cloud_path.startswith(f"{self._get_protocol()}://"):
                raise ValueError(f"Invalid path. It should start with '{self._get_protocol()}://'.")

            with self.fs.open(cloud_path, "rb") as f:
                content = f.read()

            if self.logger:
                self.logger.info(f"Downloaded file from {cloud_path}")

            return content
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error downloading file from {self.cloud_provider.upper()}: {str(e)}")
            raise

    def list_pdf_files(self, cloud_folder: str) -> List[str]:
        try:
            if not cloud_folder.startswith(f"{self._get_protocol()}://"):
                raise ValueError(f"Invalid folder path. It should start with '{self._get_protocol()}://'.")

            pdf_files = [
                f"{self._get_protocol()}://{file}" for file in self.fs.glob(cloud_folder + "/*.pdf")
            ]

            if self.logger:
                self.logger.info(f"Found {len(pdf_files)} PDF files in {cloud_folder}")

            return pdf_files

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error listing PDF files from {self.cloud_provider.upper()}: {str(e)}")
            raise

    def move_to_archive(self, cloud_folder: str, archive_folder: str) -> List[str]:
        try:
            if not cloud_folder.startswith(f"{self._get_protocol()}://"):
                raise ValueError(f"Invalid folder path. It should start with '{self._get_protocol()}://'.")

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            archive_folder_with_timestamp = f"{archive_folder.rstrip('/')}/{timestamp}"

            moved_files = []
            files_to_move = self.fs.glob(cloud_folder + "/*")

            for source_file in files_to_move:
                dest_file = f"{archive_folder_with_timestamp}/{source_file.split('/')[-1]}"
                self.fs.move(f"{self._get_protocol()}://{source_file}", f"{self._get_protocol()}://{dest_file}")

                moved_files.append(f"{self._get_protocol()}://{dest_file}")

                if self.logger:
                    self.logger.info(f"Moved {source_file} to archive: {dest_file}")

            return moved_files

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error moving files in {self.cloud_provider.upper()}: {str(e)}")
            raise

    def _get_protocol(self):
        if self.cloud_provider == "gcp":
            return "gs"
        elif self.cloud_provider == "aws":
            return "s3"
        else:
            raise ValueError("Invalid cloud_provider.")
