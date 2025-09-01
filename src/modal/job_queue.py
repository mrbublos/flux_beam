import modal

from src.app.logger import Logger

image = (modal.Image
         .debian_slim()
         .pip_install("fastapi[standard]")
         .add_local_python_source("src")
         )
app = modal.App("cv-job-queue", image=image)

ACTION_TO_APP = {
    "train": "Train",
    "inference": "Inference",
    "file": "FileManipulator",
    "check_style": "CheckStyle",
}

logger = Logger(__name__)

@app.function(image=image)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="cv", requires_proxy_auth=True)
def fastapi_app():
    import fastapi

    web_app = fastapi.FastAPI()

    @web_app.post("/submit")
    def submit_job(action: str, data: dict):
        logger.info(f"Submitting job for {action} {data['user_id']}")

        app_name = ACTION_TO_APP[action]
        cls = modal.Cls.from_name(app_name, app_name)
        call = cls.run.spawn(data)
        return call.object_id

    @web_app.post("/status/{call_id}")
    def get_job_result(call_id):
        logger.info(f"Getting job result for {call_id}")

        function_call = modal.FunctionCall.from_id(call_id)
        try:
            result = function_call.get(timeout=0)
        except modal.exception.OutputExpiredError:
            result = {"status": "expired"}
        except TimeoutError:
            result = {"status": "pending"}
        except Exception as e:
            result = {"status": "error", "error": str(e)}
        return result

    return web_app

