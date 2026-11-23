from typing import Any


def _repository(image: str) -> str:
    name = image.split("@", 1)[0]
    return name.rsplit(":", 1)[0] if ":" in name.rsplit("/", 1)[-1] else name


class _PinnedCollection:
    def __init__(self, owner: "PinnedDockerClient", collection: Any) -> None:
        self.owner = owner
        self.collection = collection

    def __getattr__(self, name: str) -> Any:
        return getattr(self.collection, name)


class _PinnedContainers(_PinnedCollection):
    def create(self, image: str, *args: Any, **kwargs: Any) -> Any:
        self.owner.check_requested_image(image)
        return self.collection.create(self.owner.image_id, *args, **kwargs)

    def run(self, image: str, *args: Any, **kwargs: Any) -> Any:
        self.owner.check_requested_image(image)
        return self.collection.run(self.owner.image_id, *args, **kwargs)


class _PinnedImages(_PinnedCollection):
    def pull(self, repository: str, tag: str | None = None, **kwargs: Any) -> Any:
        self.owner.check_requested_image(repository)
        return self.collection.pull(self.owner.reference, **kwargs)


class PinnedDockerClient:
    def __init__(self, client: Any, reference: str, image_id: str) -> None:
        self.client = client
        self.reference = reference
        self.image_id = image_id
        self.containers = _PinnedContainers(self, client.containers)
        self.images = _PinnedImages(self, client.images)

    def check_requested_image(self, image: str) -> None:
        if image not in {self.reference, self.image_id} and _repository(image) != _repository(
            self.reference
        ):
            raise ValueError(
                "Native provider requested an image outside its verified runtime profile"
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)
