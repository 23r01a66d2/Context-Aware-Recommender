import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import SchemaMappingPage from '../pages/admin/SchemaMappingPage';
import { ClientProvider } from '../context/ClientContext';
import { api } from '../services/api';
import { CANONICAL_SCHEMA_ROLES } from '../constants/schemaRoles';

vi.mock('../services/api', () => ({
  api: {
    getHealth: vi.fn(),
    getClients: vi.fn(),
    getSchema: vi.fn(),
    saveSchema: vi.fn(),
  },
}));

function renderWithProviders(ui) {
  return render(
    <BrowserRouter>
      <ClientProvider>{ui}</ClientProvider>
    </BrowserRouter>
  );
}

describe('SchemaMappingPage Frontend Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getHealth.mockResolvedValue({ status: 'ok', service: 'context-aware-recommender' });
    api.getClients.mockResolvedValue([
      { client_id: 'demo_ecommerce', name: 'Demo E-Commerce', active_model_version: 'v1' },
    ]);
  });

  // Test G: Schema mapping contains all canonical roles including BEHAVIOR_FEATURE and IGNORE
  it('G: Schema mapping contains all 8 canonical roles including BEHAVIOR_FEATURE and IGNORE', async () => {
    api.getSchema.mockResolvedValueOnce({
      client_id: 'demo_ecommerce',
      column_mappings: {
        customer_id: 'USER_ID',
        product_id: 'ITEM_ID',
        purchase_date: 'TIMESTAMP',
        purchased: 'TARGET',
        historical_clicks: 'BEHAVIOR_FEATURE',
        category: 'CONTENT_FEATURE',
        device: 'CONTEXT_FEATURE',
        cart_revenue: 'IGNORE',
      },
      column_types: {
        customer_id: 'id',
        product_id: 'id',
        purchase_date: 'datetime',
        purchased: 'numeric',
        historical_clicks: 'numeric',
        category: 'categorical',
        device: 'categorical',
        cart_revenue: 'numeric',
      },
      capabilities: {
        recommendation_compatible: true,
        supervised_training: true,
        temporal_training: true,
        cold_start_history: true,
      },
      is_valid: true,
      cold_start_threshold: 3,
      version: 1,
    });

    renderWithProviders(<SchemaMappingPage />);

    await waitFor(() => {
      expect(screen.getByText('customer_id')).toBeInTheDocument();
    });

    // Check that CANONICAL_SCHEMA_ROLES array contains all required values
    const roleValues = CANONICAL_SCHEMA_ROLES.map((r) => r.value);
    expect(roleValues).toContain('USER_ID');
    expect(roleValues).toContain('ITEM_ID');
    expect(roleValues).toContain('TIMESTAMP');
    expect(roleValues).toContain('TARGET');
    expect(roleValues).toContain('BEHAVIOR_FEATURE');
    expect(roleValues).toContain('CONTENT_FEATURE');
    expect(roleValues).toContain('CONTEXT_FEATURE');
    expect(roleValues).toContain('IGNORE');
    expect(roleValues).not.toContain('DROP'); // Must NOT use DROP

    // Check that select dropdown options render canonical roles
    const selects = screen.getAllByRole('combobox');
    const roleOptions = Array.from(selects[0].querySelectorAll('option')).map((o) => o.value);
    expect(roleOptions).toContain('BEHAVIOR_FEATURE');
    expect(roleOptions).toContain('IGNORE');
  });
});
