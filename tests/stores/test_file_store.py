"""Tests for SqlAlchemyFileStore."""

from __future__ import annotations

from agent_plane.stores.file_store.sqlalchemy_store import SqlAlchemyFileStore


def test_create_and_get(file_store: SqlAlchemyFileStore) -> None:
    f = file_store.create(filename="data.csv", bytes=1024)
    assert f.id.startswith("file_")
    assert f.filename == "data.csv"
    assert f.bytes == 1024

    fetched = file_store.get(f.id)
    assert fetched is not None
    assert fetched.filename == "data.csv"


def test_get_nonexistent(file_store: SqlAlchemyFileStore) -> None:
    assert file_store.get("file_nonexistent") is None


def test_create_with_content_type(file_store: SqlAlchemyFileStore) -> None:
    f = file_store.create(
        filename="img.png",
        bytes=2048,
        content_type="image/png",
    )
    assert f.content_type == "image/png"


def test_delete(file_store: SqlAlchemyFileStore) -> None:
    f = file_store.create(filename="temp.txt", bytes=10)
    assert file_store.delete(f.id) is True
    assert file_store.get(f.id) is None
    assert file_store.delete(f.id) is False


def test_list_pagination(file_store: SqlAlchemyFileStore) -> None:
    for i in range(4):
        file_store.create(filename=f"f{i}.txt", bytes=i)

    page1 = file_store.list(limit=2)
    assert len(page1.data) == 2
    assert page1.has_more is True

    page2 = file_store.list(limit=2, after=page1.last_id)
    assert len(page2.data) == 2
    assert page2.has_more is False


def test_list_order_asc(file_store: SqlAlchemyFileStore) -> None:
    for i in range(3):
        file_store.create(filename=f"f{i}.txt", bytes=i)
    page_desc = file_store.list(order="desc")
    page_asc = file_store.list(order="asc")
    assert [f.id for f in page_asc.data] == list(reversed([f.id for f in page_desc.data]))


def test_list_asc_with_after_cursor(file_store: SqlAlchemyFileStore) -> None:
    for i in range(5):
        file_store.create(filename=f"f{i}.txt", bytes=i)

    page1 = file_store.list(limit=2, order="asc")
    page2 = file_store.list(limit=2, order="asc", after=page1.last_id)
    page3 = file_store.list(limit=2, order="asc", after=page2.last_id)

    all_ids = [f.id for f in page1.data + page2.data + page3.data]
    full_asc = file_store.list(limit=100, order="asc")
    assert all_ids == [f.id for f in full_asc.data]
