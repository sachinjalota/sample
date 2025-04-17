from fastapi import UploadFile, File, HTTPException
from pydantic import BaseModel, Field, root_validator, field_validator
import base64
from src.config import UPLOAD_FILE_LIMIT
from enum import Enum
from typing import Optional


class FileTypeEnum(str, Enum):
    pdf = 'application/pdf'
    csv = 'text/csv'
    excel = 'application/vnd.ms-excel'
    word = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'


class FileExtensionEnum(str, Enum):
    pdf = '.pdf'
    csv = '.csv'
    excel = '.xls'
    word = '.docx'


class UploadObjectPayload(BaseModel):
    file_base64: str = Field(..., description="The base64-encoded file data")
    file_name: str = Field(..., description="File name of the given file")
    mime_type: str = Field(..., description="The MIME type of the uploaded file")
    usecase_name: Optional[str] = Field(
        "khoj", description="its useful to arrange the data into cloud storage"
    )

    @root_validator(pre=True)
    def validate_file_extension_and_mime_type(cls, values):
        file_name = values.get('file_name')
        mime_type = values.get('mime_type')
        print(mime_type, file_name)

        if not mime_type:
            raise ValueError("MIME type is required")

        # Check if mime_type is supported
        if mime_type not in [ft.value for ft in FileTypeEnum]:
            raise ValueError(f"Unsupported MIME type: {mime_type}")

        # Get corresponding file extension
        file_type_mapping = {
            FileTypeEnum.pdf.value: FileExtensionEnum.pdf.value,
            FileTypeEnum.csv.value: FileExtensionEnum.csv.value,
            FileTypeEnum.excel.value: FileExtensionEnum.excel.value,
            FileTypeEnum.word.value: FileExtensionEnum.word.value
        }

        expected_extension = file_type_mapping[mime_type]

        if file_name and not file_name.lower().endswith(expected_extension):
            raise ValueError(f"Invalid file extension. Expected {expected_extension} for MIME type {mime_type}.")

        # # Convert Word MIME type to PDF
        # if mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        #     values['mime_type'] = 'application/pdf'

        return values

    @field_validator('file_base64')
    def validate_file_size(cls, v):
        decoded_file = base64.b64decode(v)
        file_size = len(decoded_file)

        if file_size > UPLOAD_FILE_LIMIT:
            raise ValueError("File size exceeds the 10MB limit.")

        return v

    def decode_file(self):
        return base64.b64decode(self.file_base64)
