import docker
import pytest
import requests


@pytest.fixture(scope="module")
def docker_image():
    client = docker.from_env()
    # Image Build
    image, _ = client.images.build(path=".", tag="modelo-ds")
    yield image
    # Test cleaning
    client.containers.prune()


def test_modelo_dockerizado(docker_image):
    client = docker.from_env()
    # Start container
    container = client.containers.run(
        "model-ds",
        ports={"8080/tcp": 8080},
        detach=True,
        remove=True,  # Remove container after test
    )

    try:
        # Try until API is disponible
        import time

        time.sleep(5)

        # API request test
        response = requests.post(
            "http://localhost:8080/predict",
            json={"feature1": 1.2, "feature2": 0.8},
        )
        assert response.status_code == 200
        assert "prediction" in response.json()

    finally:
        container.stop()
