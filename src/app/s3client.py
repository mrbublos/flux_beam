import os
import logging
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class S3Client:
    """A client for interacting with Amazon S3.
    """

    def __init__(self):
        """Initializes the S3 client using environment variables."""
        self.bucket = os.getenv('S3_BUCKET')
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('S3_ENDPOINT'),
            aws_access_key_id=os.getenv('S3_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('S3_SECRET_KEY'),
            region_name=os.getenv('S3_REGION'),
        )
        log.info(f"S3 client initialized. Endpoint: {self.s3_client.meta.endpoint_url}, Bucket: {self.bucket}")

    def upload_file(self, data: bytes, object_name: str) -> bool:
        """Upload a file to an S3 bucket.

        :param file_name: Path to the file to upload.
        :param bucket: The name of the bucket to upload to.
        :param object_name: The S3 object name. If not specified, file_name is used.
        :return: True if the file was uploaded successfully, else False.
        """
        if object_name is None or bytearray is None:
            raise Exception("Missing required parameters")

        try:
            log.info(f"Uploading {object_name} to {self.bucket}/{object_name}")
            self.s3_client.put_object(Body=data, Bucket=self.bucket, Key=object_name)
            log.info(f"Successfully uploaded {object_name} to {self.bucket}/{object_name}")
            return True
        except ClientError as e:
            log.error(f"Failed to upload {object_name}: {e}")
            return False
        except FileNotFoundError:
            log.error(f"The file {object_name} was not found.")
            return False

    def download_file(self, object_name: str) -> bytes:
        """Download a file from an S3 bucket.

        :param object_name: The S3 object name.
        :param file_name: The local path to save the downloaded file. If not specified, object_name is used.
        :return: True if the file was downloaded successfully, else False.
        """
        if object_name is None:
            raise Exception("Missing required parameters")

        try:
            log.info(f"Downloading {object_name} from bucket {self.bucket} to {object_name}")
            result = self.s3_client.get_object(Bucket=self.bucket, Key=object_name)
            log.info(f"Successfully downloaded {object_name}")
            return result['Body'].read()
        except ClientError as e:
            log.error(f"Failed to download {object_name}: {e}")
            raise Exception(f"Failed to download {object_name}")

    def remove_object(self, object_name: str) -> bool:
        """Remove an object from an S3 bucket.

        :param bucket: The name of the bucket.
        :param object_name: The S3 object name to remove.
        :return: True if the object was removed successfully, else False.
        """
        try:
            log.info(f"Removing {object_name} from bucket {self.bucket}")
            self.s3_client.delete_object(Bucket=self.bucket, Key=object_name)
            log.info(f"Successfully removed {object_name} from bucket {self.bucket}")
            return True
        except ClientError as e:
            log.error(f"Failed to remove {object_name}: {e}")
            return False


if __name__ == '__main__':
    # This is an example of how to use the S3Client.
    # Before running, make sure to set the required environment variables:
    # export AWS_ACCESS_KEY_ID=your_key
    # export AWS_SECRET_ACCESS_KEY=your_secret
    # export AWS_REGION=your_region
    # export S3_BUCKET=your_bucket_name
    # Optional for S3-compatible storage:
    # export S3_ENDPOINT_URL=http://localhost:9000

    log.info("Running S3Client example...")
    s3_bucket = os.getenv('S3_BUCKET')

    if not s3_bucket:
        log.error("S3_BUCKET environment variable not set. Exiting example.")
    else:
        client = S3Client()

        # Create a dummy file for upload
        with open("sample.txt", "w") as f:
            f.write("This is a test file for S3 upload.")

        # Upload the file
        upload_success = client.upload_file("sample.txt", s3_bucket, "sample-remote.txt")

        if upload_success:
            # Download the file
            client.download_file(s3_bucket, "sample-remote.txt", "sample-downloaded.txt")

            # Remove the object from S3
            remove_success = client.remove_object(s3_bucket, "sample-remote.txt")
            if remove_success:
                log.info("Cleaned up S3 object.")

        # Clean up local files
        os.remove("sample.txt")
        if os.path.exists("sample-downloaded.txt"):
            os.remove("sample-downloaded.txt")

    log.info("S3Client example finished.")
