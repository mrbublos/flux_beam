import requests
import time
import os

# --- Configuration ---
# Replace with your actual API endpoint URLs
START_ENDPOINT_URL = "https://api.runpod.ai/v2/egj55t7fnu8xh3/run"
# Using .format() for job_id insertion
STATUS_ENDPOINT_URL_TEMPLATE = "https://api.runpod.ai/v2/egj55t7fnu8xh3/status/{job_id}"

# It's recommended to use environment variables for sensitive data like API keys.
# You can set this in your shell: export API_KEY='your_api_key'
API_KEY = os.environ.get("API_KEY")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

POLLING_INTERVAL_SECONDS = 10


def start_job(payload):
    """Calls the start endpoint to initiate a job."""
    print(f"Sending request to start job at: {START_ENDPOINT_URL}")
    try:
        response = requests.post(START_ENDPOINT_URL, json=payload, headers=HEADERS)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()
        job_id = response_data.get('id')

        if not job_id:
            print("Error: 'job_id' not found in the response.")
            return None

        print(f"Job started successfully. Job ID: {job_id}")
        return job_id

    except requests.exceptions.RequestException as e:
        print(f"Error starting job: {e}")
        return None

def poll_job_status(job_id):
    """Polls the status endpoint until the job is completed."""
    status_url = STATUS_ENDPOINT_URL_TEMPLATE.format(job_id=job_id)
    print(f"Polling status from: {status_url}")

    while True:
        try:
            response = requests.get(status_url, headers=HEADERS)
            response.raise_for_status()

            status_data = response.json()
            status = status_data.get('status')

            print(f"Current job status: {status}")

            if status == 'COMPLETED':
                print("Job completed successfully!")
                print("Final response:", status_data)
                break
            elif status in ['FAILED', 'ERROR', 'CANCELLED']:
                print("Job failed.")
                print("Final response:", status_data)
                break

            # Wait for the specified interval before polling again
            time.sleep(POLLING_INTERVAL_SECONDS)

        except requests.exceptions.RequestException as e:
            print(f"Error polling job status: {e}")
            print(f"Retrying in {POLLING_INTERVAL_SECONDS} seconds...")
            time.sleep(POLLING_INTERVAL_SECONDS)


if __name__ == "__main__":
    # Example payload to start the job. Modify this as needed.
    job_payload = {
        "input": {
            "prompt": "lady with a cat",
            "user_id": "test_arina",
            "num_steps": 30,
            "style_link": "https://civitai.com/api/download/models/1943313?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        }
    }

    job_id = start_job(job_payload)

    if job_id:
        poll_job_status(job_id)
