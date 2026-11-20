import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from admin.parse_pdf import run_pipeline
from admin.seed_qdrant import seed_all_manifests
from app.retrieval.bm25 import BM25Retriever
from shared.settings import general_settings


class _FakeDoc:
    def save_as_json(self, path: Path) -> None:
        path.write_text('{"doc": "full"}', encoding="utf-8")


class _FakeChunker:
    def __init__(self, **_kwargs: object) -> None:
        self.last_report = {"chunks_total": 1}
        self.last_doc = _FakeDoc()

    def process_pdf(self, _pdf_path: Path, _doc_id: str) -> tuple[list, str]:
        return [], "# Parsed document"


class AdminManifestOutputTests(unittest.TestCase):
    def test_parser_keeps_required_manifest_at_root_and_artifacts_in_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with patch("admin.parse_pdf.FastDoclingChunker", _FakeChunker):
                _, manifest_path = run_pipeline(
                    pdf_path=output_dir / "source.pdf",
                    doc_id="sample",
                    out_dir=output_dir,
                )

            metadata_dir = output_dir / general_settings.MANIFEST_METADATA_SUBDIR
            self.assertEqual(manifest_path, output_dir / "sample_manifest.json")
            self.assertEqual(list(output_dir.glob("*.json")), [manifest_path])
            self.assertTrue((metadata_dir / "sample_document.md").is_file())
            self.assertTrue((metadata_dir / "sample_quality.json").is_file())
            self.assertTrue((metadata_dir / "sample_docling.json").is_file())

    def test_bulk_seeding_only_reads_root_manifest_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_dir = Path(temp_dir)
            metadata_dir = manifest_dir / general_settings.MANIFEST_METADATA_SUBDIR
            metadata_dir.mkdir()
            manifest_path = manifest_dir / "sample_manifest.json"
            manifest_path.write_text("[]", encoding="utf-8")
            (metadata_dir / "sample_quality.json").write_text("{}", encoding="utf-8")
            (metadata_dir / "sample_docling.json").write_text("{}", encoding="utf-8")

            with patch("admin.seed_qdrant.seed_database", return_value=0) as seed:
                seed_all_manifests(manifest_dir=manifest_dir)

        self.assertEqual(seed.call_count, 1)
        self.assertEqual(Path(seed.call_args.kwargs["manifest_path"]), manifest_path)

    def test_bm25_fallback_ignores_metadata_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_dir = Path(temp_dir) / "manifests"
            metadata_dir = manifest_dir / general_settings.MANIFEST_METADATA_SUBDIR
            metadata_dir.mkdir(parents=True)
            (metadata_dir / "quality.json").write_text(
                json.dumps([{"content": "not a manifest"}]), encoding="utf-8"
            )

            with patch("app.retrieval.bm25.settings.BM25_MANIFEST_PATH", str(manifest_dir / "missing")):
                retriever = BM25Retriever(data_dir=temp_dir)

        self.assertEqual(retriever.valid_chunks, [])
