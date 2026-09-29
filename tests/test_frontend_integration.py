"""
Phase 6 Integration Tests: React Frontend Serving and API Interoperability.
Verifies that FastAPI serves the production React build (SPA routing)
and maintains full backward compatibility for all /api endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from backend.main import app


class TestFrontendIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_endpoint_still_valid(self):
        """Confirms /api/health returns exact specification alongside frontend."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "context-aware-recommender")

    def test_02_spa_root_serves_html(self):
        """Confirms root / serves the React index.html."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("<div id=\"root\"></div>", res.text)
        self.assertIn("Context-Aware Recommender", res.text)
        self.assertIn("Multi-Modal", res.text)

    def test_03_spa_client_side_routes_serve_html(self):
        """Confirms deep client-side routes return index.html for React Router."""
        routes = [
            "/admin",
            "/admin/clients",
            "/admin/clients/demo_ecommerce/dataset",
            "/admin/clients/demo_ecommerce/schema",
            "/admin/clients/demo_ecommerce/training",
            "/admin/clients/demo_ecommerce/models",
            "/admin/clients/demo_ecommerce/analytics",
            "/admin/clients/demo_ecommerce/feedback",
            "/recommend"
        ]
        for route in routes:
            res = self.client.get(route)
            self.assertEqual(res.status_code, 200, f"Route {route} failed")
            self.assertIn("<div id=\"root\"></div>", res.text)

    def test_04_api_404_preserved_under_api_prefix(self):
        """Confirms non-existent API routes return JSON 404, not HTML."""
        res = self.client.get("/api/non_existent_route_404")
        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.headers.get("content-type"), "application/json")


if __name__ == "__main__":
    unittest.main()
