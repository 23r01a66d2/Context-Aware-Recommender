import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import VisualAnalyticsPage from '../pages/admin/VisualAnalyticsPage';
import { ClientProvider } from '../context/ClientContext';
import { ThemeProvider } from '../context/ThemeContext';
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
    getVisualDatasets: vi.fn(),
    uploadVisualDataset: vi.fn(),
    getDatasetProfile: vi.fn(),
    getChartData: vi.fn(),
    getVisualRecommendations: vi.fn(),
    recordVisualInteraction: vi.fn(),
    getVisualInteractions: vi.fn(),
    getVisualAnalytics: vi.fn(),
    getClients: vi.fn(),
    getUserHistoryStatus: vi.fn(),
    getUserHistory: vi.fn(),
    recordUserHistory: vi.fn(),
  },
}));


function renderWithProviders(ui) {
  return render(
    <BrowserRouter>
      <ThemeProvider>
        <ClientProvider>{ui}</ClientProvider>
      </ThemeProvider>
    </BrowserRouter>
  );
}

describe('VisualAnalyticsPage Tests', () => {
  const mockDatasets = [
    {
      id: 'demo_ecommerce',
      name: 'Indian E-Commerce Benchmark',
      filename: 'Ecommerce.csv',
      is_canonical: true,
      size_mb: 5.94,
    },
  ];

  const mockProfile = {
    dimensions: {
      total_rows: 25000,
      total_columns: 29,
      memory_mb: 5.94,
      numeric_columns_count: 14,
      categorical_columns_count: 15,
      total_missing_values: 0,
    },
    filename: 'Ecommerce.csv',
    columns: [
      { name: 'product_category', dtype: 'categorical', unique_count: 8, null_count: 0, null_pct: 0, sample_values: ['0', '1'], stats: { top_categories: [{ category: '0', pct: 25 }] } },
      { name: 'unit_price', dtype: 'numeric', unique_count: 899, null_count: 0, null_pct: 0, sample_values: ['199', '249'], stats: { mean: 250, min: 10, max: 999, skewness: 0.12 } },
      { name: 'discount_amount', dtype: 'numeric', unique_count: 450, null_count: 0, null_pct: 0, sample_values: ['15', '20'], stats: { mean: 45, min: 0, max: 200, skewness: 0.45 } },
    ],
    numeric_columns: ['unit_price', 'discount_amount'],
    categorical_columns: ['product_category'],
    missing_summary: [],
    sample_records: [
      { product_category: '0', unit_price: 199, discount_amount: 15 },
    ],
  };

  const mockChartData = {
    chart_type: 'bar',
    x_column: 'product_category',
    y_column: 'count',
    total_records: 25000,
    data: [
      { label: 'Category 0', value: 5000 },
      { label: 'Category 1', value: 4200 },
    ],
  };

  const mockRecommendations = {
    score_source: 'neural_model',
    model_version: 'v1',
    is_cold_start: false,
    history_count: 6,
    gating_weights: { behavior: 0.35, content: 0.45, context: 0.20 },
    recommendations: [
      {
        candidate_id: 'cand_hist_unit_price',
        title: 'Unit Price Spread & Density',
        chart_type: 'histogram',
        x_col: 'unit_price',
        y_col: null,
        recommendation_score: 0.88,
        score_source: 'neural_model',
        score_label: 'Neural Personalization (v1)',
        explanation: 'Ranked by v1 neural gating based on historical category affinity.',
      },
      {
        candidate_id: 'cand_scatter_price_discount',
        title: 'Price vs Discount Relationship',
        chart_type: 'scatter',
        x_col: 'unit_price',
        y_col: 'discount_amount',
        recommendation_score: 0.82,
        score_source: 'neural_model',
        score_label: 'Neural Personalization (v1)',
        explanation: 'Strong correlation identified between unit price and discount amount.',
      },
    ],
  };

  const mockAnalytics = {
    data_origin: 'measured_production_data',
    summary: {
      total_events: 12,
      impressions: 8,
      clicks: 4,
      click_through_rate: 0.50,
      feedback_helpful: 3,
      feedback_unhelpful: 0,
      helpful_ratio: 1.0,
    },
    cohort_performance: {
      cold_start: { threshold: '< 3', impressions: 3, clicks: 1, click_through_rate: 0.33 },
      warm_start: { threshold: '>= 3', impressions: 5, clicks: 3, click_through_rate: 0.60 },
    },
    chart_type_distribution: [
      { chart_type: 'Bar', count: 5 },
      { chart_type: 'Histogram', count: 3 },
    ],
    offline_benchmark_reference: {
      overall_ndcg_at_5: 0.145,
      cold_start_mitigation_lift_pct: 28.5,
    },
  };

  beforeEach(() => {
    vi.clearAllMocks();
    api.getClients.mockResolvedValue([{ client_id: 'demo_ecommerce', name: 'Demo E-Commerce' }]);
    api.getVisualDatasets.mockResolvedValue(mockDatasets);
    api.getDatasetProfile.mockResolvedValue(mockProfile);
    api.getChartData.mockResolvedValue(mockChartData);
    api.getVisualRecommendations.mockResolvedValue(mockRecommendations);
    api.recordVisualInteraction.mockResolvedValue({ id: 1, status: 'recorded' });
    api.getVisualInteractions.mockResolvedValue([]);
    api.getVisualAnalytics.mockResolvedValue(mockAnalytics);
    api.getUserHistoryStatus.mockResolvedValue({
      client_id: 'demo_ecommerce',
      user_id: '1',
      status: 'warm_start',
      is_cold_start: false,
      history_count: 6,
      last_active: '2026-10-12T10:00:00Z',
      recent_datasets: ['demo_ecommerce']
    });
    api.getUserHistory.mockResolvedValue({
      client_id: 'demo_ecommerce',
      user_id: '1',
      status: 'warm_start',
      total_records: 1,
      history: [
        { event_id: 'evt_1', entry_type: 'chart_view', dataset_id: 'demo_ecommerce', timestamp: '2026-10-12T10:00:00Z' }
      ]
    });
  });


  it('renders page header and essential controls', async () => {
    renderWithProviders(<VisualAnalyticsPage />);

    expect(screen.getByText(/Context-Aware Visual Analytics/i)).toBeInTheDocument();
    expect(screen.getByText(/Interactive Analytics & Recommendations/i)).toBeInTheDocument();
    expect(screen.getByText(/Upload CSV/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText(/25,000/i)).toBeInTheDocument();
      expect(screen.getByText(/29 cols · 5.94 MB/i)).toBeInTheDocument();
    });
  });

  it('renders recommendation candidates with scores and explanations', async () => {
    renderWithProviders(<VisualAnalyticsPage />);

    await waitFor(() => {
      expect(screen.getByText(/Recommended Visualizations/i)).toBeInTheDocument();
      expect(screen.getByText(/Unit Price Spread & Density/i)).toBeInTheDocument();
      expect(screen.getByText(/88%/i)).toBeInTheDocument();
      expect(screen.getByText(/Price vs Discount Relationship/i)).toBeInTheDocument();
      expect(screen.getByText(/82%/i)).toBeInTheDocument();
    });
  });

  it('allows switching to Dataset Profiling & Schema tab', async () => {
    const user = userEvent.setup();
    renderWithProviders(<VisualAnalyticsPage />);

    await waitFor(() => {
      expect(screen.getByText(/Dataset Profiling & Schema/i)).toBeInTheDocument();
    });

    const datasetTabBtn = screen.getByText(/Dataset Profiling & Schema/i);
    await user.click(datasetTabBtn);

    await waitFor(() => {
      expect(screen.getByText(/Dataset Schema & Statistical Properties/i)).toBeInTheDocument();
      expect(screen.getAllByText(/product_category/i).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/unit_price/i).length).toBeGreaterThan(0);
    });
  });

  it('allows switching to Model Analytics tab and displays measured metrics', async () => {
    const user = userEvent.setup();
    renderWithProviders(<VisualAnalyticsPage />);

    await waitFor(() => {
      expect(screen.getByText(/Model Analytics & Interaction Metrics/i)).toBeInTheDocument();
    });

    const analyticsTabBtn = screen.getByText(/Model Analytics & Interaction Metrics/i);
    await user.click(analyticsTabBtn);

    await waitFor(() => {
      expect(screen.getByText(/Live Measured Production Data/i)).toBeInTheDocument();
      expect(screen.getByText(/50.0%/i)).toBeInTheDocument(); // CTR
      expect(screen.getByText(/Cold-Start vs Returning-User Performance/i)).toBeInTheDocument();
    });
  });

  it('clicking Apply Chart calls chart selection and records interaction', async () => {
    const user = userEvent.setup();
    renderWithProviders(<VisualAnalyticsPage />);

    await waitFor(() => {
      expect(screen.getByText(/Unit Price Spread & Density/i)).toBeInTheDocument();
    });

    const renderButtons = screen.getAllByText(/Render This Chart/i);
    await user.click(renderButtons[0]);

    await waitFor(() => {
      expect(api.recordVisualInteraction).toHaveBeenCalledWith(
        expect.objectContaining({
          event_type: 'recommendation_click',
          chart_id: 'cand_hist_unit_price',
        })
      );
    });
  });

  it('renders Recommendation Model Selector and fetches recommendations when changed', async () => {
    const user = userEvent.setup();
    renderWithProviders(<VisualAnalyticsPage />);

    await waitFor(() => {
      expect(screen.getByLabelText(/Recommendation Model/i)).toBeInTheDocument();
    });

    const modelSelect = screen.getByLabelText(/Recommendation Model/i);
    expect(modelSelect.value).toBe('mlvis');

    await user.selectOptions(modelSelect, 'statistical');

    await waitFor(() => {
      expect(api.getVisualRecommendations).toHaveBeenCalledWith(
        expect.objectContaining({
          model_type: 'statistical',
        })
      );
    });
  });

  it('handles CSV upload, refreshes dataset list, selects uploaded dataset, and shows success banner', async () => {
    const user = userEvent.setup();
    const updatedDatasets = [
      ...mockDatasets,
      {
        id: 'upload_va_user_sales_data.csv',
        name: 'sales_data.csv (Uploaded: va_user)',
        filename: 'sales_data.csv',
        client_id: 'va_user',
        is_canonical: false,
        size_mb: 0.15,
      },
    ];

    api.uploadVisualDataset.mockResolvedValueOnce({
      dataset_id: 'upload_va_user_sales_data.csv',
      filename: 'sales_data.csv',
      row_count: 50,
      column_count: 4,
    });
    api.getVisualDatasets.mockResolvedValueOnce(mockDatasets).mockResolvedValueOnce(updatedDatasets);

    const { container } = renderWithProviders(<VisualAnalyticsPage />);

    await waitFor(() => {
      expect(screen.getByText(/Upload CSV/i)).toBeInTheDocument();
    });

    const fileInput = container.querySelector('input[type="file"]');
    expect(fileInput).not.toBeNull();

    const fakeFile = new File(['store,sales\nA,100\nB,200\n'], 'sales_data.csv', { type: 'text/csv' });
    await user.upload(fileInput, fakeFile);

    await waitFor(() => {
      expect(api.uploadVisualDataset).toHaveBeenCalledWith(fakeFile, 'demo_ecommerce', '1');
      expect(screen.getByText(/Successfully uploaded "sales_data.csv"/i)).toBeInTheDocument();
      expect(screen.getByDisplayValue(/sales_data.csv/i)).toBeInTheDocument();
    });

  });

  it('displays error message when CSV upload fails', async () => {
    const user = userEvent.setup();
    api.uploadVisualDataset.mockRejectedValueOnce(new Error('Invalid or corrupted CSV file'));

    const { container } = renderWithProviders(<VisualAnalyticsPage />);

    await waitFor(() => {
      expect(screen.getByText(/Upload CSV/i)).toBeInTheDocument();
    });

    const fileInput = container.querySelector('input[type="file"]');
    const fakeFile = new File(['bad_content'], 'corrupt.csv', { type: 'text/csv' });
    await user.upload(fileInput, fakeFile);

    await waitFor(() => {
      expect(screen.getByText(/Upload error: Invalid or corrupted CSV file/i)).toBeInTheDocument();
    });
  });

  it('renders all extended chart types and allows selecting Pie, Area, and Box charts', async () => {
    const user = userEvent.setup();
    renderWithProviders(<VisualAnalyticsPage />);

    await waitFor(() => {
      expect(screen.getByText(/25,000/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /^Bar$/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /^Area$/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /^Stacked Bar$/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /^Bubble$/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /^Box Plot$/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /^Pie$/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /^Donut$/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /^Treemap$/i })).toBeInTheDocument();
    });

    const pieBtn = screen.getByRole('button', { name: /^Pie$/i });
    await user.click(pieBtn);

    await waitFor(() => {
      expect(api.getChartData).toHaveBeenCalledWith(
        expect.objectContaining({
          chart_type: 'pie',
        })
      );
    });
  });

  it('renders MLVis v2 Candidate Release option in Model Engine dropdown and fetches recommendations', async () => {
    const user = userEvent.setup();
    renderWithProviders(<VisualAnalyticsPage />);

    await waitFor(() => {
      expect(screen.getByLabelText(/Recommendation Model/i)).toBeInTheDocument();
    });

    const modelSelect = screen.getByLabelText(/Recommendation Model/i);
    expect(screen.getByRole('option', { name: /MLVis Recommender v2 \(Candidate Release, v2.0.0\)/i })).toBeInTheDocument();

    await user.selectOptions(modelSelect, 'mlvis_v2');

    await waitFor(() => {
      expect(api.getVisualRecommendations).toHaveBeenCalledWith(
        expect.objectContaining({
          model_type: 'mlvis_v2',
        })
      );
    });
  });

  it('renders User ID step and displays Cold Start vs Warm Start status', async () => {
    api.getUserHistoryStatus.mockResolvedValueOnce({
      client_id: 'demo_ecommerce',
      user_id: 'new_cold_user',
      status: 'cold_start',
      is_cold_start: true,
      history_count: 0,
      last_active: null,
      recent_datasets: []
    });

    renderWithProviders(<VisualAnalyticsPage />);

    expect(screen.getByText(/Active User Identification & Profile/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Enter User ID/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByTestId('user-status-cold')).toBeInTheDocument();
      expect(screen.getByText(/Cold-Start Profile Detected/i)).toBeInTheDocument();
    });
  });

  it('renders history-based personalization label and unsuitable preferred chart alert banner', async () => {
    api.getVisualRecommendations.mockResolvedValueOnce({
      score_source: 'mlvis_model',
      model_version: 'mlvis_v1.0.0',
      is_cold_start: false,
      has_chart_preference: true,
      personalization_mode: 'warm_personalized',
      unsuitable_preferred_charts: [
        {
          preferred_chart: 'line',
          reason: 'Dataset lacks sequential or temporal progression columns',
          suggested_alternatives: ['bar', 'histogram']
        }
      ],
      recommendations: [
        {
          candidate_id: 'cand_bar_cat',
          title: 'Category Volume',
          chart_type: 'bar',
          recommendation_score: 0.92,
          base_model_confidence: 0.82,
          history_personalization_boost: 0.10,
          preference_applied: true,
          personalization_label: 'Suggested based on your previous bar selections',
          score_source: 'mlvis_model',
          score_label: 'MLVis + Boost',
          explanation: 'Predicted canonical bar with +10% boost from past selections'
        },
        {
          candidate_id: 'cand_hist_price',
          title: 'Price Histogram',
          chart_type: 'histogram',
          recommendation_score: 0.80,
          base_model_confidence: 0.80,
          history_personalization_boost: 0.0,
          preference_applied: false,
          personalization_label: null,
          score_source: 'mlvis_model',
          score_label: 'MLVis Base',
          explanation: 'Standard MLVis prediction'
        }
      ]
    });

    renderWithProviders(<VisualAnalyticsPage />);

    // Unsuitable chart callout banner should be rendered
    await waitFor(() => {
      expect(screen.getByText(/Your preferred line chart is not suitable for this dataset/i)).toBeInTheDocument();
      expect(screen.getByText(/Dataset lacks sequential or temporal progression columns/i)).toBeInTheDocument();
    });

    // Personalization label on the boosted chart
    expect(screen.getByText(/Suggested based on your previous bar selections/i)).toBeInTheDocument();

    // Verify unboosted chart does not have personalization label
    expect(screen.queryByText(/Suggested based on your previous histogram selections/i)).toBeNull();
  });
});


