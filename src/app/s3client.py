import os
import logging
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class S3Client:
    """A client for interacting with Amazon S3.

    This client's configuration is sourced from environment variables:
    - AWS_ACCESS_KEY_ID: Your AWS access key ID.
    - AWS_SECRET_ACCESS_KEY: Your AWS secret access key.
    - AWS_SESSION_TOKEN: Your AWS session token (optional).
    - AWS_REGION: The AWS region to use (e.g., 'us-east-1').
    - S3_ENDPOINT_URL: The custom endpoint URL for S3-compatible storage (optional).
    """

    def __init__(self):
        """Initializes the S3 client using environment variables."""
        endpoint_url = os.getenv('S3_ENDPOINT_URL')
        self.s3_client = boto3.client('s3', endpoint_url=endpoint_url)
        log.info(f"S3 client initialized. Endpoint URL: {endpoint_url or 'Default AWS'}")

    def upload_file(self, file_name: str, bucket: str, object_name: str = None) -> bool:
        """Upload a file to an S3 bucket.

        :param file_name: Path to the file to upload.
        :param bucket: The name of the bucket to upload to.
        :param object_name: The S3 object name. If not specified, file_name is used.
        :return: True if the file was uploaded successfully, else False.
        """
        if object_name is None:
            object_name = os.path.basename(file_name)

        try:
            log.info(f"Uploading {file_name} to {bucket}/{object_name}")
            self.s3_client.upload_file(file_name, bucket, object_name)
            log.info(f"Successfully uploaded {file_name} to {bucket}/{object_name}")
            return True
        except ClientError as e:
            log.error(f"Failed to upload {file_name}: {e}")
            return False
        except FileNotFoundError:
            log.error(f"The file {file_name} was not found.")
            return False

    def download_file(self, bucket: str, object_name: str, file_name: str = None) -> bool:
        """Download a file from an S3 bucket.

        :param bucket: The name of the bucket to download from.
        :param object_name: The S3 object name.
        :param file_name: The local path to save the downloaded file. If not specified, object_name is used.
        :return: True if the file was downloaded successfully, else False.
        """
        if file_name is None:
            file_name = object_name

        try:
            log.info(f"Downloading {object_name} from bucket {bucket} to {file_name}")
            self.s3_client.download_file(bucket, object_name, file_name)
            log.info(f"Successfully downloaded {object_name} to {file_name}")
            return True
        except ClientError as e:
            log.error(f"Failed to download {object_name}: {e}")
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

        # Clean up local files
        os.remove("sample.txt")
        if os.path.exists("sample-downloaded.txt"):
            os.remove("sample-downloaded.txt")

    log.info("S3Client example finished.")