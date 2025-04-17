import os
import subprocess
import tempfile
import src.config as config
import time
from io import BytesIO
import base64
from fastapi import HTTPException
from src.models.upload_object_payload import UploadObjectPayload
import uuid
import datetime
import shutil
import traceback


class FileUploadService:
    def __init__(self, storage_service, logger):
        self.storage_service = storage_service
        self.logger = logger

    async def upload_object(self, payload: UploadObjectPayload):
        temp_path = None
        try:
            # Decode the base64 file and prepare the file-like object
            try:
                decoded_file = base64.b64decode(payload.file_base64)
                file_like_object = BytesIO(decoded_file)
                file_like_object.seek(0)
            except Exception as e:
                self.logger.error(f"Base64 decoding failed: {str(e)}")
                raise HTTPException(status_code=400, detail="Invalid file format provided.")

            current_date = datetime.datetime.now()
            year = current_date.strftime("%Y")
            month = current_date.strftime("%m")
            day = current_date.strftime("%d")
            bucket_name = config.UPLOAD_BUCKET_NAME
            folder_name = config.UPLOAD_FOLDER_NAME
            extension = payload.file_name.split('.')[1]
            full_path = f"{folder_name}/{payload.usecase_name}/{year}/{month}/{day}/{payload.file_name}"
            if extension in ['docx', 'doc']:
                full_path = f"{folder_name}/{payload.usecase_name}/{year}/{month}/{day}/{payload.file_name.split('.')[0]}.pdf"
                temp_path = self.create_temp_file(file_like_object.getvalue(), payload.file_name)
                file_like_object = self.convert_to_pdf(file_like_object.getvalue(), temp_path)

            self.logger.info(f"Attempting to upload {payload.file_name} to {bucket_name}/{full_path}")
            response = self.storage_service.upload_object(file_like_object, bucket_name, full_path)
            unique_id = uuid.uuid4().hex
            self.logger.info(f"upload object : {response} , unique id : {unique_id}")
            return {
                "unique_id": unique_id,
                "object_path": response
            }
        except FileNotFoundError:
            self.logger.error("The specified file could not be found in the storage service.")
            raise HTTPException(status_code=404, detail="File not found.")
        except PermissionError:
            self.logger.error("Permission denied during file upload.")
            raise HTTPException(status_code=403, detail="Permission denied.")
        except Exception as e:
            self.logger.error(f"File upload failed: {str(e)}\nTraceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
        finally:
            if temp_path:
                self.clean_temp_file(temp_path)

    def create_temp_file(self, file_content: bytes, file_name: str) -> str:
        # Create temp file with specified name
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file_name)

        try:
            # Write content to temp file
            with open(temp_path, 'wb') as temp_file:
                temp_file.write(file_content)
            self.logger.info(f"Temp file creates for: {file_name} in {temp_path}")
            return temp_path
        except Exception as e:
            self.logger.error(f"Error creating temp file: {str(e)}\nTraceback: {traceback.format_exc()}")
            self.logger.error(f"Error While creating temp file : {file_name} in {temp_path} error:{str(e)}")
            # Clean up temp file if writing fails
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return ''

    def download_file(self, cloud_object_path):
        try:
            self.logger.info(f"Downloading file  {cloud_object_path}")
            file_content = self.storage_service.download_object(cloud_object_path)
            file_name = cloud_object_path.split('/')[-1]
            temp_path = self.create_temp_file(file_content, file_name)
            return temp_path
        except Exception as e:
            self.logger.error(f"Error creating temp file: {str(e)}\nTraceback: {traceback.format_exc()}")
            return ''

    def clean_temp_file(self, temp_path: str):
        """Delete temporary file."""
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                self.logger.info(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                self.logger.error(f"Error cleaning temp file {temp_path}: {str(e)}")

    def convert_to_pdf(self, file_like_object: bytes, file_name: str):
        tempfile_path = self.create_temp_file(file_like_object, file_name)
        try:
            # Convert docx/doc to PDF using LibreOffice
            output_dir = os.path.dirname(tempfile_path)
            output_filename = os.path.splitext(file_name)[0] + '.pdf'
            output_path = os.path.join(output_dir, output_filename)

            # Run LibreOffice conversion command
            conversion_command = [
                '/usr/bin/libreoffice25.2',
                '--headless',
                '--convert-to', 'pdf',
                '--outdir', output_dir,
                tempfile_path
            ]

            process = subprocess.Popen(
                conversion_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                self.logger.error(f"PDF conversion failed: {stderr.decode()}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to convert document to PDF"
                )

            # Update file details for upload
            file_name = output_filename
            with open(output_path, 'rb') as pdf_file:
                file_like_object = BytesIO(pdf_file.read())
                file_like_object.seek(0)
            return file_like_object
            # Cleanup temp files
            # self.delete_temp_file(tempfile_path)
            # self.delete_temp_file(output_path)

        except Exception as e:
            self.logger.error(f"Document conversion failed: {str(e)}")
            self.logger.error(f"Document conversion failed: {traceback.format_exc()}")
            if 'tempfile_path' in locals():
                self.delete_temp_file(tempfile_path)
            if 'output_path' in locals():
                self.delete_temp_file(output_path)
            raise HTTPException(
                status_code=500,
                detail=f"Document conversion failed: {str(e)}"
            )
