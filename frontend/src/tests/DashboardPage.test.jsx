import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import DashboardPage from '../pages/admin/DashboardPage';
import DatasetClientsShowcase from '../components/DatasetClientsShowcase';
import { ClientProvider } from '../context/ClientContext';
import { api } from '../services/api';

if (typeof window !== 'undefined' && !window.ResizeObserver) {
  window.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}
if (typeof global !== 'undefined' && !global.ResizeObserver) {
  global.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

vi.mock('../services/api', () => ({
  api: {
    getHealth: vi.fn(),
    getClients: vi.fn(),
    getClientsOverview: vi.fn(),
    getFeedbackLog: vi.fn(),
    getModels: vi.fn(),
    getDataset: vi.fn(),
    activateModel: vi.fn(),
  },
}));

function renderWithProviders(ui) {
  return render(
    <BrowserRouter>
      <ClientProvider>{ui}</ClientProvider>
    </BrowserRouter>
  );
}

describe('DashboardPage & DatasetClientsShowcase Tests', () => {
  const mockClients = [
    {
      client_id: 'demo_ecommerce',
      name: 'Indian E-Commerce Benchmark',
      active_model_version: 'v1',
      is_system: true,
    },
    {
      client_id: 'movielens_dataset',
      name: 'MovieLens 100k Benchmark',
      active_model_version: 'v2',
      is_system: false,
    },
  ];

  const mockOverviews = [
    {
      client_id: 'demo_ecommerce',
      name: 'Indian E-Commerce Benchmark',
      description: 'Canonical reference dataset and research benchmark model.',
      domain: 'Products / E-Commerce',
      dataset_name: 'Indian E-Commerce Customer Behavior & Purchase',
      user_label: 'Customers (customer_id)',
      item_label: 'Products (product_id)',
      users_count: 8442,
      items_count: 899,
      interactions_count: 25000,
      interaction_types: ['Purchases', 'Views', 'Clicks'],
      available_models: ['v1 (Active)'],
      active_model_version: 'v1',
      supports_context: true,
      context_features: ['device_type', 'user_type'],
      supports_implicit_feedback: true,
      is_system: true,
    },
    {
      client_id: 'movielens_dataset',
      name: 'MovieLens 100k Benchmark',
      description: 'Standard MovieLens research dataset for movies domain.',
      domain: 'Movies',
      dataset_name: 'MovieLens 100k (ml-latest-small)',
      user_label: 'Users (user_id)',
      item_label: 'Movies (movie_id / titles)',
      users_count: 610,
      items_count: 9724,
      interactions_count: 100836,
      interaction_types: ['Ratings (0.5 - 5.0 Stars)', 'Clicks'],
      available_models: ['v1', 'v2 (Active)'],
      active_model_version: 'v2',
      supports_context: true,
      context_features: ['interaction_datetime', 'genre_action'],
      supports_implicit_feedback: true,
      is_system: false,
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    api.getHealth.mockResolvedValue({ status: 'ok', service: 'context-aware-recommender' });
    api.getClients.mockResolvedValue(mockClients);
    api.getClientsOverview.mockResolvedValue(mockOverviews);
    api.getFeedbackLog.mockResolvedValue([]);
    api.getModels.mockResolvedValue([{ version_tag: 'v1', is_active: true }]);
    api.getDataset.mockResolvedValue({ row_count: 25000, column_count: 29 });
  });

  it('renders DashboardPage with DatasetClientsShowcase without runtime errors', async () => {
    renderWithProviders(<DashboardPage />);

    // Verify Dashboard Banner
    expect(screen.getByText(/Multi-Client Recommendation Platform/i)).toBeInTheDocument();

    // Verify DatasetClientsShowcase elements rendered on Dashboard
    await waitFor(() => {
      expect(screen.getByText(/Available Datasets & Recommendation Clients/i)).toBeInTheDocument();
      expect(screen.getAllByText('Indian E-Commerce Benchmark').length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('MovieLens 100k Benchmark')).toBeInTheDocument();
    });
  });

  it('allows selecting another client domain in DatasetClientsShowcase', async () => {
    const user = userEvent.setup();
    renderWithProviders(<DatasetClientsShowcase />);

    await waitFor(() => {
      expect(screen.getByText('MovieLens 100k Benchmark')).toBeInTheDocument();
    });

    const movielensCard = screen.getByText('MovieLens 100k Benchmark').closest('div');
    await user.click(movielensCard);

    await waitFor(() => {
      expect(screen.getByText(/Switched active domain to movielens_dataset/i)).toBeInTheDocument();
    });
  });
});
