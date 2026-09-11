"""Client wrapper for interacting with Label Studio SDK."""

from pathlib import Path
from typing import Sequence, Optional, Any, Union
from label_studio_sdk import Client
from mmsar.annotation.label_studio_config import generate_label_studio_config


class LabelStudioManager:
    """Manager wrapping Label Studio SDK for project setup and task imports."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str = "dummy-api-key",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client: Optional[Client] = None

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(url=self.base_url, api_key=self.api_key)
        return self._client

    def create_project(
        self,
        title: str = "MMSAR Annotation Project",
        label_config: Optional[str] = None,
    ) -> Any:
        """Create a new labeling project with the given configuration.

        Args:
            title: Project title.
            label_config: XML string for project configuration.

        Returns:
            Created Project instance from SDK.
        """
        if label_config is None:
            label_config = generate_label_studio_config()

        project = self.client.start_project(
            title=title,
            label_config=label_config.strip(),
        )
        return project

    def import_image_tasks(
        self,
        project_id: int,
        image_paths: Sequence[Union[str, Path]],
    ) -> list[int]:
        """Import a collection of image paths as annotation tasks.

        Args:
            project_id: ID of the project to import into.
            image_paths: Sequence of image file paths or URLs.

        Returns:
            List of imported task IDs.
        """
        project = self.client.get_project(project_id)
        tasks = [{"data": {"image": str(p)}} for p in image_paths]
        task_ids = project.import_tasks(tasks)
        return task_ids
