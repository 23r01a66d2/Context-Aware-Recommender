import React, { useState, useEffect, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useClient } from '../../context/ClientContext';
import { useTheme } from '../../context/ThemeContext';
import { api } from '../../services/api';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  PieChart,
  Pie,
  AreaChart,
  Area,
  Treemap,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Cell
} from 'recharts';
import {
  BarChart3,
  LineChart as LineChartIcon,
  ScatterChart as ScatterIcon,
  PieChart as PieChartIcon,
  Table,
  UploadCloud,
  Sparkles,
  Sliders,
  Filter,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  ThumbsUp,
  ThumbsDown,
  Info,
  HelpCircle,
  Layers,
  ChevronRight,
  TrendingUp,
  Eye,
  Activity,
  Compass,
  FileSpreadsheet,
  CircleDot,
  Box as BoxIcon,
  LayoutGrid
} from 'lucide-react';
import UserIdentityStep from '../../components/UserIdentityStep';


export default function VisualAnalyticsPage({
  embeddedInModelWorkspace = false,
  lockedModel = null,
  modelName = null,
  supportsVisualizations = true,
  hideHeader = false,
  headerTitle = null,
  headerSubtitle = null
} = {}) {
  const { clientId } = useParams();
  const { selectedClientId, selectClient } = useClient();
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === 'dark';

  const activeClientId = clientId || selectedClientId || 'demo_ecommerce';

  // Persistent session identifier
  const [sessionId] = useState(() => {
    let sid = sessionStorage.getItem('va_session_id');
    if (!sid) {
      sid = 'va_session_' + Math.random().toString(36).substring(2, 10);
      sessionStorage.setItem('va_session_id', sid);
    }
    return sid;
  });

  // State: Datasets
  const [datasets, setDatasets] = useState([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState('demo_ecommerce');
  const [selectedUserId, setSelectedUserId] = useState('1'); // Customer 1 in demo_ecommerce
  const [selectedModel, setSelectedModel] = useState(() => lockedModel || 'mlvis'); // mlvis, context_aware, statistical, custom

  // State: Dataset Profiling
  const [profile, setProfile] = useState(null);
  const [profileLoading, setProfileLoading] = useState(true);
  const [profileError, setProfileError] = useState(null);

  // State: Active Chart Configuration
  const [chartType, setChartType] = useState('bar'); // bar, line, histogram, scatter, heatmap
  const [xColumn, setXColumn] = useState('');
  const [yColumn, setYColumn] = useState('');
  const [aggregation, setAggregation] = useState('count');
  const [chartData, setChartData] = useState(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [chartError, setChartError] = useState(null);

  // State: Filters
  const [selectedFilterCategory, setSelectedFilterCategory] = useState('');
  const [minPriceFilter, setMinPriceFilter] = useState('');
  const [maxPriceFilter, setMaxPriceFilter] = useState('');

  // State: Recommendations
  const [recommendations, setRecommendations] = useState([]);
  const [recMeta, setRecMeta] = useState(null);
  const [recsLoading, setRecsLoading] = useState(false);

  // State: Analytics & Event Logging
  const [analytics, setAnalytics] = useState(null);
  const [interactions, setInteractions] = useState([]);
  const [feedbackStatus, setFeedbackStatus] = useState({}); // candidateId -> 'helpful' | 'unhelpful'
  const [activeTab, setActiveTab] = useState('explorer'); // explorer, dataset, analytics

  // Upload state
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(null);
  const fileInputRef = React.useRef(null);

  // 1. Load available datasets on mount
  useEffect(() => {
    async function loadDatasets() {
      try {
        const list = await api.getVisualDatasets();
        const safeList = Array.isArray(list) ? list : [];
        setDatasets(safeList);
        if (safeList.length > 0 && !selectedDatasetId) {
          setSelectedDatasetId(safeList[0].id);
        }
      } catch (err) {
        console.error('Failed to load visual datasets:', err);
        setDatasets([]);
      }
    }
    loadDatasets();
  }, []);

  // 2. Fetch dataset profile when selected dataset changes
  useEffect(() => {
    async function fetchProfile() {
      if (!selectedDatasetId) return;
      setProfileLoading(true);
      setProfileError(null);
      try {
        const prof = await api.getDatasetProfile(selectedDatasetId, activeClientId);
        setProfile(prof);

        // Auto-select initial columns
        if (prof.columns && prof.columns.length > 0) {
          const numCols = prof.numeric_columns || [];
          const catCols = prof.categorical_columns || [];

          if (catCols.length > 0) {
            setXColumn(catCols[0]);
          } else if (numCols.length > 0) {
            setXColumn(numCols[0]);
          }

          if (numCols.length > 1) {
            setYColumn(numCols[1]);
          } else if (numCols.length > 0) {
            setYColumn(numCols[0]);
          }
        }
      } catch (err) {
        setProfileError(err.message || 'Failed to profile dataset.');
      } finally {
        setProfileLoading(false);
      }
    }
    fetchProfile();
  }, [selectedDatasetId, activeClientId]);

  // 3. Fetch Recommendations whenever dataset, user, or active chart changes
  const fetchRecommendations = async () => {
    if (!selectedDatasetId || !profile) return;
    setRecsLoading(true);
    try {
      const activeFilters = {};
      if (selectedFilterCategory) activeFilters.product_category = { op: 'eq', val: selectedFilterCategory };
      if (minPriceFilter) activeFilters.unit_price = { op: 'gte', val: parseFloat(minPriceFilter) };

      const res = await api.getVisualRecommendations({
        dataset_id: selectedDatasetId,
        client_id: activeClientId,
        user_id: selectedUserId,
        model_type: selectedModel,
        session_context: {
          session_id: sessionId,
          current_chart_type: chartType,
          selected_columns: [xColumn, yColumn].filter(Boolean),
          active_filters: activeFilters,
        },
        top_k: 6,
      });

      setRecommendations(res.recommendations || []);
      setRecMeta({
        score_source: res.score_source,
        model_version: res.model_version,
        model_type: res.model_type,
        is_cold_start: res.is_cold_start,
        gating_weights: res.gating_weights,
        history_count: res.history_count,
        personalization_mode: res.personalization_mode,
        has_chart_preference: res.has_chart_preference,
        unsuitable_preferred_charts: res.unsuitable_preferred_charts || [],
      });

      // Log impression for recommendations
      if (res.recommendations && res.recommendations.length > 0) {
        Promise.resolve(
          api.recordVisualInteraction({
            session_id: sessionId,
            event_type: 'chart_impression',
            client_id: activeClientId,
            user_id: selectedUserId,
            chart_id: res.recommendations[0].candidate_id,
            chart_type: res.recommendations[0].chart_type,
            details: { recommendation_count: res.recommendations.length, source: res.score_source, model_type: selectedModel },
          })
        ).catch(() => {});
      }
    } catch (err) {
      console.error('Error fetching visual recommendations:', err);
    } finally {
      setRecsLoading(false);
    }
  };

  useEffect(() => {
    if (profile) {
      fetchRecommendations();
    }
  }, [profile, selectedUserId, chartType, selectedModel]);

  // 4. Fetch Chart Data whenever configuration or filters change
  const fetchChartData = async () => {
    if (!profile || !xColumn) return;
    setChartLoading(true);
    setChartError(null);
    try {
      const activeFilters = {};
      if (selectedFilterCategory) {
        const catCol = profile.categorical_columns?.[0] || 'product_category';
        activeFilters[catCol] = { op: 'eq', val: selectedFilterCategory };
      }
      if (minPriceFilter || maxPriceFilter) {
        const priceCol = profile.numeric_columns?.[0] || 'unit_price';
        if (minPriceFilter && maxPriceFilter) {
          activeFilters[priceCol] = { op: 'between', val: [parseFloat(minPriceFilter), parseFloat(maxPriceFilter)] };
        } else if (minPriceFilter) {
          activeFilters[priceCol] = { op: 'gte', val: parseFloat(minPriceFilter) };
        } else if (maxPriceFilter) {
          activeFilters[priceCol] = { op: 'lte', val: parseFloat(maxPriceFilter) };
        }
      }

      const res = await api.getChartData({
        dataset_id: selectedDatasetId,
        client_id: activeClientId,
        chart_type: chartType,
        x_column: xColumn,
        y_column: ([
          'scatter',
          'line',
          'bubble',
          'area',
          'box',
          'stacked_bar',
          'grouped_bar',
          'pie',
          'donut',
          'treemap'
        ].includes(chartType) || (chartType === 'bar' && aggregation !== 'count'))
          ? (yColumn || undefined)
          : undefined,
        aggregation: aggregation,
        filters: Object.keys(activeFilters).length > 0 ? activeFilters : undefined,
      });

      setChartData(res);
    } catch (err) {
      setChartError(err.message || 'Failed to render chart.');
    } finally {
      setChartLoading(false);
    }
  };

  useEffect(() => {
    fetchChartData();
  }, [selectedDatasetId, chartType, xColumn, yColumn, aggregation, selectedFilterCategory, minPriceFilter, maxPriceFilter]);

  // 5. Fetch Analytics & Audit History
  const fetchAnalytics = async () => {
    try {
      const [analyticsRes, historyRes] = await Promise.all([
        api.getVisualAnalytics(activeClientId),
        api.getVisualInteractions(sessionId, activeClientId),
      ]);
      setAnalytics(analyticsRes);
      setInteractions(historyRes || []);
    } catch (err) {
      console.error('Failed to load visual analytics:', err);
    }
  };

  useEffect(() => {
    fetchAnalytics();
  }, [activeClientId, sessionId]);

  // Handle applying a recommendation to the live chart explorer
  const applyRecommendation = (rec) => {
    setChartType(rec.chart_type);
    if (rec.x_col) setXColumn(rec.x_col);
    if (rec.y_col) setYColumn(rec.y_col);
    if (rec.aggregation && rec.aggregation !== 'none') setAggregation(rec.aggregation);

    // Record click event
    api.recordVisualInteraction({
      session_id: sessionId,
      event_type: 'recommendation_click',
      client_id: activeClientId,
      user_id: selectedUserId,
      chart_id: rec.candidate_id,
      chart_type: rec.chart_type,
      details: { score: rec.recommendation_score, score_source: rec.score_source },
    }).then(fetchAnalytics).catch(() => {});
  };

  // Handle feedback (helpful / unhelpful)
  const submitFeedback = async (rec, rating) => {
    const isHelpful = rating === 'helpful';
    setFeedbackStatus((prev) => ({ ...prev, [rec.candidate_id]: rating }));

    try {
      await api.recordVisualInteraction({
        session_id: sessionId,
        event_type: isHelpful ? 'helpful_feedback' : 'unhelpful_feedback',
        client_id: activeClientId,
        user_id: selectedUserId,
        chart_id: rec.candidate_id,
        chart_type: rec.chart_type,
        details: { rating, score: rec.recommendation_score, score_source: rec.score_source },
      });
      fetchAnalytics();
      fetchRecommendations(); // Refresh to reflect feedback adaptation
    } catch (err) {
      console.error('Feedback submission failed:', err);
    }
  };

  // Handle CSV file upload
  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);
    try {
      const res = await api.uploadVisualDataset(file, activeClientId, selectedUserId);
      // Reload datasets and select the newly uploaded file
      const updatedList = await api.getVisualDatasets();
      setDatasets(updatedList);
      
      const newDatasetId = res.dataset_id || res.filename;
      setSelectedDatasetId(newDatasetId);
      setUploadSuccess(`Successfully uploaded "${res.filename}" (${res.row_count} rows, ${res.column_count} columns).`);
      setActiveTab('explorer');

      // Clear success banner after 6 seconds
      setTimeout(() => {
        setUploadSuccess(null);
      }, 6000);
    } catch (err) {
      setUploadError(err.message || 'CSV upload failed.');
    } finally {
      setUploading(false);
      // Reset input element value so selecting the exact same file again works
      if (e.target) {
        e.target.value = '';
      }
    }
  };

  // Synchronize lockedModel if provided
  useEffect(() => {
    if (lockedModel) {
      setSelectedModel(lockedModel);
    }
  }, [lockedModel]);

  // If model does not support visual recommendations, render clear explanation
  if (!supportsVisualizations) {
    return (
      <div className="p-8 rounded-2xl dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 text-center space-y-4 max-w-2xl mx-auto my-8">
        <div className="w-12 h-12 rounded-2xl bg-amber-500/10 text-amber-500 flex items-center justify-center mx-auto">
          <AlertCircle className="w-6 h-6" />
        </div>
        <h3 className="text-lg font-bold dark:text-coolwhite text-slate-900">
          Visualization Recommendations Unsupported
        </h3>
        <p className="text-xs dark:text-muted text-slate-500 leading-relaxed">
          {modelName || 'This model'} is designed specifically for <span className="font-semibold text-teal-500">item ranking and catalog personalization</span>, and does not support visualization recommendation or chart synthesis.
        </p>
        <div className="p-4 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 text-xs text-left">
          <div className="font-semibold dark:text-coolwhite text-slate-800 mb-1">Recommended Action:</div>
          <p className="text-muted text-[11px]">
            To explore dataset profiling, interactive charts, and intelligent chart recommendations, use the dedicated <strong className="text-teal-500">MLVis Neural Visualization Recommender</strong>.
          </p>
        </div>
        <div className="pt-2">
          <Link
            to="/admin/models/mlvis/workspace?tab=visualizations"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-teal-500 hover:bg-teal-400 text-navy-950 font-bold text-xs transition-all shadow-sm"
          >
            <Sparkles className="w-4 h-4" />
            Switch to MLVis Visualization Workspace
          </Link>
        </div>
      </div>
    );
  }

  // Colors for charts
  const colors = ['#14b8a6', '#0ea5e9', '#8b5cf6', '#f59e0b', '#ec4899', '#10b981', '#6366f1', '#f97316'];

  return (
    <div className="space-y-6 pb-12">
      {/* Top Banner & Header */}
      {!embeddedInModelWorkspace && !hideHeader && (
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b dark:border-navy-800 border-slate-200 pb-5">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/30">
                Interactive Analytics & Recommendations
              </span>
              <span className="text-xs text-muted">Session: {sessionId.slice(0, 15)}</span>
            </div>
            <h1 className="text-2xl font-bold dark:text-coolwhite text-slate-900 tracking-tight flex items-center gap-2">
              <LineChartIcon className="w-6 h-6 text-teal-500" />
              {headerTitle || 'Context-Aware Visual Analytics'}
            </h1>
            <p className="text-xs text-muted mt-1 max-w-3xl">
              {headerSubtitle || 'Explore datasets with interactive charts and receive intelligent visualization suggestions powered by dataset characteristics, user behavior, and multimodal fusion.'}
            </p>
          </div>
        </div>
      )}

      {/* User Identification Step: Cold vs Warm Start Detection & Persistent History */}
      <UserIdentityStep
        clientId={activeClientId}
        userId={selectedUserId}
        onUserIdChange={(newUid) => setSelectedUserId(newUid)}
        showHistoryDrawer={true}
      />

      {/* Global Controls: Dataset, Model Engine, User Profiles */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-50 dark:bg-navy-950/60 p-3 rounded-2xl border dark:border-navy-800 border-slate-200">
        <div className="flex flex-wrap items-center gap-3">
          {/* Dataset Selector */}
          <div className="flex items-center gap-2 bg-slate-100 dark:bg-navy-900 border dark:border-navy-800 border-slate-200 px-3 py-1.5 rounded-xl">
            <FileSpreadsheet className="w-4 h-4 text-teal-500" />
            <div className="text-left">
              <div className="text-[10px] uppercase font-bold text-muted">Dataset</div>
              <select
                aria-label="Dataset"
                value={selectedDatasetId}
                onChange={(e) => setSelectedDatasetId(e.target.value)}
                className="bg-transparent font-medium text-xs dark:text-coolwhite text-slate-800 focus:outline-none cursor-pointer"
              >
                {(datasets || []).map((d) => (
                  <option key={d.id} value={d.id} className="dark:bg-navy-900 bg-white">
                    {d.name} {d.is_canonical ? '(Benchmark)' : ''}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Model Selector Dropdown (or Locked Engine Badge) */}
          {lockedModel ? (
            <div className="flex items-center gap-2 bg-teal-500/10 border border-teal-500/30 px-3 py-1.5 rounded-xl">
              <Sparkles className="w-4 h-4 text-teal-500" />
              <div className="text-left">
                <div className="text-[10px] uppercase font-bold text-teal-600 dark:text-teal-400">Model Engine</div>
                <div className="font-bold text-xs text-teal-700 dark:text-teal-300">
                  {lockedModel === 'mlvis' ? 'MLVis Recommender (Active)' : lockedModel}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-2 bg-slate-100 dark:bg-navy-900 border dark:border-navy-800 border-slate-200 px-3 py-1.5 rounded-xl">
              <Sparkles className="w-4 h-4 text-teal-500" />
              <div className="text-left">
                <div className="text-[10px] uppercase font-bold text-muted">Model Engine</div>
                <select
                  aria-label="Recommendation Model"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="bg-transparent font-medium text-xs dark:text-coolwhite text-slate-800 focus:outline-none cursor-pointer"
                >
                  <option value="mlvis" className="dark:bg-navy-900 bg-white">
                    MLVis Recommender (Primary, v1.0.0 Production)
                  </option>
                  <option value="mlvis_v2" className="dark:bg-navy-900 bg-white">
                    MLVis Recommender v2 (Candidate Release, v2.0.0)
                  </option>
                  <option value="context_aware" className="dark:bg-navy-900 bg-white">
                    Context-Aware Recommender (Two-Tower)
                  </option>
                  <option value="statistical" className="dark:bg-navy-900 bg-white">
                    Cold-Start Statistical Fallback
                  </option>
                  <option value="custom" className="dark:bg-navy-900 bg-white">
                    Custom Dataset Mode
                  </option>
                </select>
              </div>
            </div>
          )}

          {/* User Profile Selector (for demonstrating cold-start vs warm behavior) */}
          <div className="flex items-center gap-2 bg-slate-100 dark:bg-navy-900 border dark:border-navy-800 border-slate-200 px-3 py-1.5 rounded-xl">
            <Compass className="w-4 h-4 text-purple-500" />
            <div className="text-left">
              <div className="text-[10px] uppercase font-bold text-muted">User Context</div>
              <select
                aria-label="User Context"
                value={selectedUserId}
                onChange={(e) => setSelectedUserId(e.target.value)}
                className="bg-transparent font-medium text-xs dark:text-coolwhite text-slate-800 focus:outline-none cursor-pointer"
              >
                <option value="new_user" className="dark:bg-navy-900 bg-white">New User (Cold-Start, N=0)</option>
                <option value="1" className="dark:bg-navy-900 bg-white">Customer 1 (Warm User, N=6)</option>
                <option value="10" className="dark:bg-navy-900 bg-white">Customer 10 (Warm User, N=4)</option>
                <option value="999" className="dark:bg-navy-900 bg-white">Customer 999 (Cold-Start, N=1)</option>
              </select>
            </div>
          </div>

          {/* Upload CSV Button */}
          <label
            className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all cursor-pointer shadow-sm select-none ${
              uploading
                ? 'bg-teal-600/70 text-white/80 cursor-not-allowed'
                : 'bg-teal-500 text-white hover:bg-teal-600 active:scale-95'
            }`}
          >
            {uploading ? (
              <RefreshCw className="w-4 h-4 animate-spin shrink-0" />
            ) : (
              <UploadCloud className="w-4 h-4 shrink-0" />
            )}
            <span>{uploading ? 'Uploading CSV...' : 'Upload CSV'}</span>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              disabled={uploading}
              onClick={(e) => {
                e.target.value = '';
              }}
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>
        </div>
      </div>

      {uploadSuccess && (
        <div className="p-3.5 rounded-xl bg-teal-500/10 border border-teal-500/30 text-teal-700 dark:text-teal-300 text-xs flex items-center justify-between gap-2 transition-all">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 shrink-0 text-teal-500" />
            <span>{uploadSuccess}</span>
          </div>
          <button
            onClick={() => setUploadSuccess(null)}
            className="text-teal-600 dark:text-teal-400 hover:text-teal-800 dark:hover:text-teal-200 text-xs font-semibold px-2 py-0.5 rounded"
          >
            Dismiss
          </button>
        </div>
      )}

      {uploadError && (
        <div className="p-3.5 rounded-xl bg-red-500/10 border border-red-500/30 text-red-600 dark:text-red-400 text-xs flex items-center justify-between gap-2 transition-all">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>Upload error: {uploadError}</span>
          </div>
          <button
            onClick={() => setUploadError(null)}
            className="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-200 text-xs font-semibold px-2 py-0.5 rounded"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="flex items-center gap-2 border-b dark:border-navy-800 border-slate-200 pb-2">
        <button
          onClick={() => setActiveTab('explorer')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
            activeTab === 'explorer'
              ? 'bg-teal-500/15 text-teal-600 dark:text-teal-300 border border-teal-500/30 shadow-sm'
              : 'text-muted hover:text-coolwhite hover:bg-navy-900/40'
          }`}
        >
          <BarChart3 className="w-4 h-4" />
          <span>Interactive Explorer & Recommendations</span>
        </button>

        <button
          onClick={() => setActiveTab('dataset')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
            activeTab === 'dataset'
              ? 'bg-teal-500/15 text-teal-600 dark:text-teal-300 border border-teal-500/30 shadow-sm'
              : 'text-muted hover:text-coolwhite hover:bg-navy-900/40'
          }`}
        >
          <Table className="w-4 h-4" />
          <span>Dataset Profiling & Schema ({profile?.dimensions?.total_rows ?? 0} rows)</span>
        </button>

        <button
          onClick={() => setActiveTab('analytics')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
            activeTab === 'analytics'
              ? 'bg-teal-500/15 text-teal-600 dark:text-teal-300 border border-teal-500/30 shadow-sm'
              : 'text-muted hover:text-coolwhite hover:bg-navy-900/40'
          }`}
        >
          <Activity className="w-4 h-4" />
          <span>Model Analytics & Interaction Metrics</span>
        </button>

        <button
          onClick={() => setActiveTab('history')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
            activeTab === 'history'
              ? 'bg-teal-500/15 text-teal-600 dark:text-teal-300 border border-teal-500/30 shadow-sm'
              : 'text-muted hover:text-coolwhite hover:bg-navy-900/40'
          }`}
        >
          <Compass className="w-4 h-4" />
          <span>Your History & Recently Recommended</span>
        </button>
      </div>


      {/* KPI Stats Strip */}
      {profile && profile.dimensions && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200">
            <div className="text-[11px] font-semibold text-muted uppercase">Rows Ingested</div>
            <div className="text-xl font-bold dark:text-coolwhite text-slate-900 mt-1">
              {profile.dimensions.total_rows?.toLocaleString?.() ?? profile.dimensions.total_rows ?? '—'}
            </div>
            <div className="text-[10px] text-muted mt-0.5">{profile.filename}</div>
          </div>

          <div className="p-4 rounded-xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200">
            <div className="text-[11px] font-semibold text-muted uppercase">Columns & Memory</div>
            <div className="text-xl font-bold dark:text-coolwhite text-slate-900 mt-1">
              {profile.dimensions.total_columns ?? '—'} cols · {profile.dimensions.memory_mb ?? '—'} MB
            </div>
            <div className="text-[10px] text-muted mt-0.5">
              {profile.dimensions.numeric_columns_count ?? 0} numeric, {profile.dimensions.categorical_columns_count ?? 0} categorical
            </div>
          </div>

          <div className="p-4 rounded-xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200">
            <div className="text-[11px] font-semibold text-muted uppercase">Data Quality</div>
            <div className="text-xl font-bold dark:text-coolwhite text-slate-900 mt-1 flex items-center gap-1.5">
              {profile.dimensions.total_missing_values === 0 ? (
                <>
                  <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                  <span className="text-emerald-500">100% Complete</span>
                </>
              ) : (
                <span className="text-amber-500">{profile.dimensions.total_missing_values ?? 0} missing</span>
              )}
            </div>
            <div className="text-[10px] text-muted mt-0.5">Verified zero missing values</div>
          </div>

          <div className="p-4 rounded-xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200">
            <div className="text-[11px] font-semibold text-muted uppercase">Recommendation Engine</div>
            <div className="text-xl font-bold dark:text-coolwhite text-slate-900 mt-1 flex items-center gap-1.5">
              {recMeta?.score_source === 'neural_model' ? (
                <span className="text-teal-400 text-sm">Two-Tower Multi-Modal ({recMeta?.model_version})</span>
              ) : (
                <span className="text-blue-400 text-sm">Cold-Start Suitability</span>
              )}
            </div>
            <div className="text-[10px] text-muted mt-0.5">
              {recMeta?.is_cold_start ? 'Cold-start mode (<3 history)' : 'Warm personalization'}
            </div>
          </div>
        </div>
      )}

      {/* TAB 1: INTERACTIVE EXPLORER & RECOMMENDATIONS */}
      {activeTab === 'explorer' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Main Visual Workspace: Controls & Interactive Chart (8 cols) */}
          <div className="lg:col-span-8 space-y-4">
            {/* Chart Control Bar */}
            <div className="p-4 rounded-xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-3">
                {/* Chart Type Selector */}
                <div className="flex flex-wrap items-center gap-1 p-1 bg-slate-100 dark:bg-navy-950 rounded-xl border dark:border-navy-800 border-slate-200">
                  {[
                    { id: 'bar', label: 'Bar', icon: BarChart3 },
                    { id: 'line', label: 'Line', icon: LineChartIcon },
                    { id: 'area', label: 'Area', icon: TrendingUp },
                    { id: 'stacked_bar', label: 'Stacked Bar', icon: Layers },
                    { id: 'histogram', label: 'Histogram', icon: BarChart3 },
                    { id: 'scatter', label: 'Scatter', icon: ScatterIcon },
                    { id: 'bubble', label: 'Bubble', icon: CircleDot },
                    { id: 'box', label: 'Box Plot', icon: BoxIcon },
                    { id: 'pie', label: 'Pie', icon: PieChartIcon },
                    { id: 'donut', label: 'Donut', icon: CircleDot },
                    { id: 'treemap', label: 'Treemap', icon: LayoutGrid },
                    { id: 'heatmap', label: 'Heatmap', icon: PieChartIcon },
                  ].map((ct) => {
                    const Icon = ct.icon;
                    return (
                      <button
                        key={ct.id}
                        onClick={() => {
                          setChartType(ct.id);
                          if (selectedUserId) {
                            api.recordVisualInteraction({
                              session_id: sessionId,
                              event_type: 'chart_selection',
                              client_id: activeClientId,
                              user_id: selectedUserId,
                              chart_type: ct.id,
                              details: { manual_selection: true }
                            }).catch(() => {});
                          }
                        }}
                        className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                          chartType === ct.id
                            ? 'bg-teal-500 text-white shadow-sm'
                            : 'text-muted hover:text-coolwhite'
                        }`}
                      >
                        <Icon className="w-3.5 h-3.5" />
                        <span>{ct.label}</span>
                      </button>
                    );
                  })}
                </div>

                {/* X-Axis Selector */}
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-bold text-muted uppercase">X Axis:</span>
                  <select
                    aria-label="X Axis"
                    value={xColumn}
                    onChange={(e) => setXColumn(e.target.value)}
                    className="bg-slate-100 dark:bg-navy-950 border dark:border-navy-800 border-slate-200 rounded-lg px-2.5 py-1 text-xs dark:text-coolwhite text-slate-800 font-medium focus:outline-none"
                  >
                    {profile?.columns?.map((c) => (
                      <option key={c.name} value={c.name}>
                        {c.name} ({c.dtype})
                      </option>
                    ))}
                  </select>
                </div>

                {/* Y-Axis Selector (when relevant) */}
                {['scatter', 'line', 'bar', 'area', 'bubble', 'box', 'treemap', 'stacked_bar', 'pie', 'donut'].includes(chartType) && (
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] font-bold text-muted uppercase">Y Axis:</span>
                    <select
                      aria-label="Y Axis"
                      value={yColumn}
                      onChange={(e) => setYColumn(e.target.value)}
                      className="bg-slate-100 dark:bg-navy-950 border dark:border-navy-800 border-slate-200 rounded-lg px-2.5 py-1 text-xs dark:text-coolwhite text-slate-800 font-medium focus:outline-none"
                    >
                      <option value="">None (Count / Default)</option>
                      {profile?.numeric_columns?.map((nc) => (
                        <option key={nc} value={nc}>{nc}</option>
                      ))}
                    </select>
                  </div>
                )}

                {/* Aggregation Selector (for bar / line / area) */}
                {['bar', 'line', 'area'].includes(chartType) && yColumn && (
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] font-bold text-muted uppercase">Agg:</span>
                    <select
                      aria-label="Aggregation"
                      value={aggregation}
                      onChange={(e) => setAggregation(e.target.value)}
                      className="bg-slate-100 dark:bg-navy-950 border dark:border-navy-800 border-slate-200 rounded-lg px-2.5 py-1 text-xs dark:text-coolwhite text-slate-800 font-medium focus:outline-none"
                    >
                      <option value="mean">Mean / Average</option>
                      <option value="sum">Sum</option>
                      <option value="min">Minimum</option>
                      <option value="max">Maximum</option>
                    </select>
                  </div>
                )}
              </div>

              {/* Filter Controls Row */}
              <div className="pt-2 border-t dark:border-navy-800 border-slate-100 flex flex-wrap items-center gap-3">
                <span className="text-[11px] font-bold text-muted uppercase flex items-center gap-1">
                  <Filter className="w-3.5 h-3.5 text-teal-500" /> Filter:
                </span>

                {/* Category Filter */}
                <select
                  aria-label="Category Filter"
                  value={selectedFilterCategory}
                  onChange={(e) => setSelectedFilterCategory(e.target.value)}
                  className="bg-slate-100 dark:bg-navy-950 border dark:border-navy-800 border-slate-200 rounded-lg px-2 py-1 text-xs dark:text-coolwhite text-slate-800 font-medium focus:outline-none"
                >
                  <option value="">All Categories</option>
                  <option value="0">Electronics (0)</option>
                  <option value="1">Fashion (1)</option>
                  <option value="2">Home & Kitchen (2)</option>
                  <option value="3">Beauty (3)</option>
                  <option value="4">Sports (4)</option>
                  <option value="5">Books (5)</option>
                </select>

                {/* Min / Max Price Filter */}
                <div className="flex items-center gap-1.5">
                  <input
                    type="number"
                    placeholder="Min Price"
                    value={minPriceFilter}
                    onChange={(e) => setMinPriceFilter(e.target.value)}
                    className="w-20 bg-slate-100 dark:bg-navy-950 border dark:border-navy-800 border-slate-200 rounded-lg px-2 py-1 text-xs dark:text-coolwhite text-slate-800 focus:outline-none"
                  />
                  <span className="text-muted text-xs">-</span>
                  <input
                    type="number"
                    placeholder="Max Price"
                    value={maxPriceFilter}
                    onChange={(e) => setMaxPriceFilter(e.target.value)}
                    className="w-20 bg-slate-100 dark:bg-navy-950 border dark:border-navy-800 border-slate-200 rounded-lg px-2 py-1 text-xs dark:text-coolwhite text-slate-800 focus:outline-none"
                  />
                </div>

                {(selectedFilterCategory || minPriceFilter || maxPriceFilter) && (
                  <button
                    onClick={() => {
                      setSelectedFilterCategory('');
                      setMinPriceFilter('');
                      setMaxPriceFilter('');
                    }}
                    className="text-[11px] text-teal-500 hover:underline font-semibold"
                  >
                    Clear Filters
                  </button>
                )}
              </div>
            </div>

            {/* Interactive Chart Canvas Card */}
            <div className="p-5 rounded-2xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200 shadow-sm relative min-h-[440px] flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="font-bold text-sm dark:text-coolwhite text-slate-900 capitalize">
                      {chartType} Visualization: {xColumn} {yColumn ? `vs ${yColumn}` : ''}
                    </h3>
                    <div className="text-[11px] text-muted">
                      {chartData?.total_records?.toLocaleString() ?? 0} matching records from {selectedDatasetId}
                    </div>
                  </div>

                  {/* Feedback Buttons on current chart */}
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => submitFeedback({ candidate_id: `custom_${chartType}_${xColumn}`, chart_type: chartType, recommendation_score: 1.0, score_source: 'user_custom' }, 'helpful')}
                      className="p-1.5 rounded-lg border dark:border-navy-800 border-slate-200 text-muted hover:text-emerald-500 hover:bg-emerald-500/10 transition-colors"
                      title="Helpful visualization"
                    >
                      <ThumbsUp className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => submitFeedback({ candidate_id: `custom_${chartType}_${xColumn}`, chart_type: chartType, recommendation_score: 1.0, score_source: 'user_custom' }, 'unhelpful')}
                      className="p-1.5 rounded-lg border dark:border-navy-800 border-slate-200 text-muted hover:text-rose-500 hover:bg-rose-500/10 transition-colors"
                      title="Not helpful"
                    >
                      <ThumbsDown className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>

                {chartError ? (
                  <div className="py-24 text-center text-xs text-red-500">
                    <AlertCircle className="w-8 h-8 mx-auto mb-2 opacity-60" />
                    {chartError}
                  </div>
                ) : chartLoading ? (
                  <div className="py-28 text-center text-xs text-muted">
                    <RefreshCw className="w-6 h-6 mx-auto mb-2 animate-spin text-teal-500" />
                    Computing visual aggregates from dataset...
                  </div>
                ) : !chartData || !chartData.data || chartData.data.length === 0 ? (
                  <div className="py-28 text-center text-xs text-muted">
                    No data points match current filter criteria.
                  </div>
                ) : (
                  <div className="h-[360px] w-full">
                    {/* BAR CHART */}
                    {chartType === 'bar' && (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData.data} margin={{ top: 10, right: 20, left: 10, bottom: 40 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1e293b' : '#f1f5f9'} />
                          <XAxis
                            dataKey="label"
                            stroke={isDark ? '#64748b' : '#94a3b8'}
                            fontSize={11}
                            angle={-25}
                            textAnchor="end"
                          />
                          <YAxis stroke={isDark ? '#64748b' : '#94a3b8'} fontSize={11} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: isDark ? '#0f172a' : '#ffffff',
                              borderColor: isDark ? '#334155' : '#cbd5e1',
                              borderRadius: '8px',
                              fontSize: '12px',
                            }}
                          />
                          <Bar dataKey="value" fill="#14b8a6" radius={[4, 4, 0, 0]}>
                            {chartData.data.map((_, idx) => (
                              <Cell key={`cell-${idx}`} fill={colors[idx % colors.length]} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    )}

                    {/* LINE CHART */}
                    {chartType === 'line' && (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData.data} margin={{ top: 10, right: 20, left: 10, bottom: 40 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1e293b' : '#f1f5f9'} />
                          <XAxis
                            dataKey="label"
                            stroke={isDark ? '#64748b' : '#94a3b8'}
                            fontSize={11}
                            angle={-25}
                            textAnchor="end"
                          />
                          <YAxis stroke={isDark ? '#64748b' : '#94a3b8'} fontSize={11} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: isDark ? '#0f172a' : '#ffffff',
                              borderColor: isDark ? '#334155' : '#cbd5e1',
                              borderRadius: '8px',
                              fontSize: '12px',
                            }}
                          />
                          <Line
                            type="monotone"
                            dataKey="value"
                            stroke="#0ea5e9"
                            strokeWidth={2.5}
                            dot={{ fill: '#0ea5e9', r: 3 }}
                            activeDot={{ r: 6 }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    )}

                    {/* HISTOGRAM */}
                    {chartType === 'histogram' && (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData.data} margin={{ top: 10, right: 20, left: 10, bottom: 40 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1e293b' : '#f1f5f9'} />
                          <XAxis
                            dataKey="label"
                            stroke={isDark ? '#64748b' : '#94a3b8'}
                            fontSize={10}
                            angle={-30}
                            textAnchor="end"
                          />
                          <YAxis stroke={isDark ? '#64748b' : '#94a3b8'} fontSize={11} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: isDark ? '#0f172a' : '#ffffff',
                              borderColor: isDark ? '#334155' : '#cbd5e1',
                              borderRadius: '8px',
                              fontSize: '12px',
                            }}
                          />
                          <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    )}

                    {/* SCATTER PLOT */}
                    {chartType === 'scatter' && (
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1e293b' : '#f1f5f9'} />
                          <XAxis
                            type="number"
                            dataKey="x"
                            name={chartData.x_column}
                            stroke={isDark ? '#64748b' : '#94a3b8'}
                            fontSize={11}
                          />
                          <YAxis
                            type="number"
                            dataKey="y"
                            name={chartData.y_column}
                            stroke={isDark ? '#64748b' : '#94a3b8'}
                            fontSize={11}
                          />
                          <Tooltip
                            cursor={{ strokeDasharray: '3 3' }}
                            contentStyle={{
                              backgroundColor: isDark ? '#0f172a' : '#ffffff',
                              borderColor: isDark ? '#334155' : '#cbd5e1',
                              borderRadius: '8px',
                              fontSize: '12px',
                            }}
                          />
                          <Scatter name="Data Points" data={chartData.data} fill="#14b8a6" fillOpacity={0.6} />
                        </ScatterChart>
                      </ResponsiveContainer>
                    )}

                    {/* CORRELATION HEATMAP */}
                    {chartType === 'heatmap' && (
                      <div className="overflow-x-auto h-full flex items-center justify-center">
                        <table className="border-collapse text-xs">
                          <thead>
                            <tr>
                              <th className="p-1"></th>
                              {chartData.columns?.map((c) => (
                                <th key={c} className="p-1.5 font-bold text-muted text-[10px] uppercase truncate max-w-[70px]">
                                  {c.replace('_', ' ')}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {chartData.matrix?.map((row, i) => (
                              <tr key={chartData.columns[i]}>
                                <td className="p-1.5 font-bold text-muted text-[10px] uppercase truncate max-w-[80px]">
                                  {chartData.columns[i].replace('_', ' ')}
                                </td>
                                {row.map((val, j) => {
                                  // Color intensity based on correlation
                                  const absVal = Math.abs(val);
                                  const isPos = val >= 0;
                                  const bgColor = isPos
                                    ? `rgba(20, 184, 166, ${Math.max(absVal, 0.1)})`
                                    : `rgba(244, 63, 94, ${Math.max(absVal, 0.1)})`;
                                  return (
                                    <td
                                      key={`${i}-${j}`}
                                      className="p-2 text-center font-mono font-semibold text-[11px] border dark:border-navy-800 border-slate-200"
                                      style={{ backgroundColor: bgColor }}
                                      title={`${chartData.columns[i]} vs ${chartData.columns[j]}: r = ${val}`}
                                    >
                                      {val.toFixed(2)}
                                    </td>
                                  );
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}

                    {/* AREA CHART */}
                    {chartType === 'area' && (
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData.data} margin={{ top: 10, right: 20, left: 10, bottom: 40 }}>
                          <defs>
                            <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                              <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.05}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1e293b' : '#f1f5f9'} />
                          <XAxis
                            dataKey="label"
                            stroke={isDark ? '#64748b' : '#94a3b8'}
                            fontSize={11}
                            angle={-25}
                            textAnchor="end"
                          />
                          <YAxis stroke={isDark ? '#64748b' : '#94a3b8'} fontSize={11} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: isDark ? '#0f172a' : '#ffffff',
                              borderColor: isDark ? '#334155' : '#cbd5e1',
                              borderRadius: '8px',
                              fontSize: '12px',
                            }}
                          />
                          <Area type="monotone" dataKey="value" stroke="#8b5cf6" strokeWidth={2.5} fillOpacity={1} fill="url(#areaGradient)" />
                        </AreaChart>
                      </ResponsiveContainer>
                    )}

                    {/* STACKED / GROUPED BAR CHART */}
                    {(chartType === 'stacked_bar' || chartType === 'grouped_bar') && (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData.data} margin={{ top: 10, right: 20, left: 10, bottom: 40 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1e293b' : '#f1f5f9'} />
                          <XAxis
                            dataKey="label"
                            stroke={isDark ? '#64748b' : '#94a3b8'}
                            fontSize={11}
                            angle={-25}
                            textAnchor="end"
                          />
                          <YAxis stroke={isDark ? '#64748b' : '#94a3b8'} fontSize={11} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: isDark ? '#0f172a' : '#ffffff',
                              borderColor: isDark ? '#334155' : '#cbd5e1',
                              borderRadius: '8px',
                              fontSize: '12px',
                            }}
                          />
                          <Legend />
                          {chartData.group_keys?.map((k, idx) => (
                            <Bar
                              key={k}
                              dataKey={k}
                              stackId={chartData.is_stacked ? 'stack' : undefined}
                              fill={colors[idx % colors.length]}
                              radius={chartData.is_stacked ? [0, 0, 0, 0] : [4, 4, 0, 0]}
                            />
                          ))}
                        </BarChart>
                      </ResponsiveContainer>
                    )}

                    {/* PIE & DONUT CHART */}
                    {(chartType === 'pie' || chartType === 'donut') && (
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                          <Tooltip
                            contentStyle={{
                              backgroundColor: isDark ? '#0f172a' : '#ffffff',
                              borderColor: isDark ? '#334155' : '#cbd5e1',
                              borderRadius: '8px',
                              fontSize: '12px',
                            }}
                            formatter={(val, name, entry) => [`${val} (${entry.payload.percentage || 0}%)`, entry.payload.label || name]}
                          />
                          <Legend />
                          <Pie
                            data={chartData.data}
                            dataKey="value"
                            nameKey="label"
                            cx="50%"
                            cy="50%"
                            outerRadius={110}
                            innerRadius={chartType === 'donut' ? 60 : 0}
                            paddingAngle={chartType === 'donut' ? 3 : 1}
                          >
                            {chartData.data?.map((_, idx) => (
                              <Cell key={`slice-${idx}`} fill={colors[idx % colors.length]} />
                            ))}
                          </Pie>
                        </PieChart>
                      </ResponsiveContainer>
                    )}

                    {/* BUBBLE CHART */}
                    {chartType === 'bubble' && (
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1e293b' : '#f1f5f9'} />
                          <XAxis
                            type="number"
                            dataKey="x"
                            name={chartData.x_column}
                            stroke={isDark ? '#64748b' : '#94a3b8'}
                            fontSize={11}
                          />
                          <YAxis
                            type="number"
                            dataKey="y"
                            name={chartData.y_column}
                            stroke={isDark ? '#64748b' : '#94a3b8'}
                            fontSize={11}
                          />
                          <ZAxis
                            type="number"
                            dataKey="size"
                            range={[60, 400]}
                            name={chartData.size_column}
                          />
                          <Tooltip
                            cursor={{ strokeDasharray: '3 3' }}
                            contentStyle={{
                              backgroundColor: isDark ? '#0f172a' : '#ffffff',
                              borderColor: isDark ? '#334155' : '#cbd5e1',
                              borderRadius: '8px',
                              fontSize: '12px',
                            }}
                            formatter={(val, name) => [val, name]}
                          />
                          <Scatter name="Bubble Points" data={chartData.data} fill="#8b5cf6" fillOpacity={0.6} />
                        </ScatterChart>
                      </ResponsiveContainer>
                    )}

                    {/* BOX PLOT (Distribution Quartiles) */}
                    {chartType === 'box' && (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData.data} margin={{ top: 10, right: 20, left: 10, bottom: 40 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1e293b' : '#f1f5f9'} />
                          <XAxis
                            dataKey="label"
                            stroke={isDark ? '#64748b' : '#94a3b8'}
                            fontSize={11}
                            angle={-25}
                            textAnchor="end"
                          />
                          <YAxis stroke={isDark ? '#64748b' : '#94a3b8'} fontSize={11} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: isDark ? '#0f172a' : '#ffffff',
                              borderColor: isDark ? '#334155' : '#cbd5e1',
                              borderRadius: '8px',
                              fontSize: '12px',
                            }}
                            formatter={(val, name, entry) => {
                              const d = entry.payload;
                              return [
                                `Min: ${d.min}, Q1: ${d.q1}, Median: ${d.median}, Q3: ${d.q3}, Max: ${d.max} (N=${d.count})`,
                                'Quartile Distribution'
                              ];
                            }}
                          />
                          <Legend />
                          <Bar dataKey="median" name="Median" fill="#7c3aed" radius={[4, 4, 0, 0]} />
                          <Bar dataKey="q3" name="Q3 (75th Percentile)" fill="#a78bfa" fillOpacity={0.6} radius={[4, 4, 0, 0]} />
                          <Bar dataKey="q1" name="Q1 (25th Percentile)" fill="#c4b5fd" fillOpacity={0.4} radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    )}

                    {/* TREEMAP */}
                    {chartType === 'treemap' && (
                      <ResponsiveContainer width="100%" height="100%">
                        <Treemap
                          data={chartData.data}
                          dataKey="size"
                          nameKey="name"
                          stroke={isDark ? '#0f172a' : '#ffffff'}
                          fill="#7c3aed"
                          content={({ root, depth, x, y, width, height, index, name, value }) => {
                            if (width < 30 || height < 20) return null;
                            const color = colors[index % colors.length];
                            return (
                              <g>
                                <rect
                                  x={x}
                                  y={y}
                                  width={width}
                                  height={height}
                                  style={{
                                    fill: color,
                                    stroke: isDark ? '#0f172a' : '#ffffff',
                                    strokeWidth: 2,
                                    opacity: 0.85
                                  }}
                                  rx={4}
                                />
                                {width > 50 && height > 30 && (
                                  <text
                                    x={x + width / 2}
                                    y={y + height / 2}
                                    textAnchor="middle"
                                    fill="#ffffff"
                                    fontSize={11}
                                    fontWeight="bold"
                                    dominantBaseline="central"
                                  >
                                    {name} ({value})
                                  </text>
                                )}
                              </g>
                            );
                          }}
                        >
                          <Tooltip
                            contentStyle={{
                              backgroundColor: isDark ? '#0f172a' : '#ffffff',
                              borderColor: isDark ? '#334155' : '#cbd5e1',
                              borderRadius: '8px',
                              fontSize: '12px',
                            }}
                          />
                        </Treemap>
                      </ResponsiveContainer>
                    )}
                  </div>
                )}
              </div>

              {/* Chart metadata strip */}
              {chartData && (
                <div className="mt-4 pt-3 border-t dark:border-navy-800 border-slate-100 flex flex-wrap items-center justify-between text-[11px] text-muted">
                  <div className="flex items-center gap-4">
                    {chartData.correlation !== undefined && (
                      <span className="font-semibold text-teal-400">
                        Pearson r: {chartData.correlation}
                      </span>
                    )}
                    {chartData.stats && (
                      <span>
                        Mean: {chartData.stats.mean} · Skewness: {chartData.stats.skewness}
                      </span>
                    )}
                    {chartData.sampled_records && (
                      <span>Subsampled {chartData.sampled_records} points for fluid rendering</span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                    <span>Live Render from Dataset</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right Rail: Visualization Recommendations (4 cols) */}
          <div className="lg:col-span-4 space-y-4">
            <div className="p-4 rounded-xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200 flex items-center justify-between">
              <div>
                <h3 className="font-bold text-xs uppercase tracking-wider dark:text-coolwhite text-slate-900 flex items-center gap-1.5">
                  <Sparkles className="w-4 h-4 text-teal-500" />
                  Recommended Visualizations
                </h3>
                <p className="text-[10px] text-muted mt-0.5">
                  Personalized based on session & data properties
                </p>
              </div>

              <button
                onClick={fetchRecommendations}
                disabled={recsLoading}
                className="p-1.5 rounded-lg border dark:border-navy-800 border-slate-200 text-muted hover:text-teal-400"
                title="Refresh recommendations"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${recsLoading ? 'animate-spin' : ''}`} />
              </button>
            </div>

            {/* Neural Gating Weights Pill (when neural model active) */}
            {recMeta?.gating_weights && (
              <div className="p-3 rounded-xl dark:bg-navy-950/80 bg-slate-50 border dark:border-navy-800 border-slate-200 text-[11px] space-y-1.5">
                <div className="flex items-center justify-between font-semibold text-muted text-[10px] uppercase">
                  <span>Adaptive Neural Gate</span>
                  <span className="text-teal-400 font-bold">{recMeta.model_version}</span>
                </div>
                <div className="grid grid-cols-3 gap-1 text-center font-mono">
                  <div className="p-1 rounded bg-teal-500/10 border border-teal-500/20 text-teal-400">
                    <div className="text-[9px] uppercase font-sans">Beh</div>
                    {Math.round(recMeta.gating_weights.behavior * 100)}%
                  </div>
                  <div className="p-1 rounded bg-blue-500/10 border border-blue-500/20 text-blue-400">
                    <div className="text-[9px] uppercase font-sans">Cont</div>
                    {Math.round(recMeta.gating_weights.content * 100)}%
                  </div>
                  <div className="p-1 rounded bg-purple-500/10 border border-purple-500/20 text-purple-400">
                    <div className="text-[9px] uppercase font-sans">Ctx</div>
                    {Math.round(recMeta.gating_weights.context * 100)}%
                  </div>
                </div>
                <div className="text-[9px] text-muted italic">
                  {recMeta.is_cold_start ? 'Cold-start active: Behavioral weight suppressed' : 'Warm-user behavioral representations active'}
                </div>
              </div>
            )}

            {/* Unsuitable Preferred Chart Callout Banner */}
            {recMeta?.unsuitable_preferred_charts && recMeta.unsuitable_preferred_charts.length > 0 && (
              <div className="space-y-2 mb-3">
                {recMeta.unsuitable_preferred_charts.map((un, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-xs text-amber-700 dark:text-amber-300 flex items-start gap-2.5"
                  >
                    <Info className="w-4 h-4 shrink-0 text-amber-500 mt-0.5" />
                    <div>
                      <div className="font-semibold">
                        Your preferred {un.preferred_chart} chart is not suitable for this dataset
                      </div>
                      <div className="text-[11px] text-muted mt-0.5">
                        {un.reason}. Showing the best statistically compatible alternatives below.
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Recommendation Candidate Cards */}
            <div className="space-y-3">
              {recommendations.map((rec) => {
                const fb = feedbackStatus[rec.candidate_id];
                const isNeural = rec.score_source === 'neural_model';

                return (
                  <div
                    key={rec.candidate_id}
                    className="p-3.5 rounded-xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200 hover:border-teal-500/40 transition-all shadow-sm space-y-2 group"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <div className="flex items-center gap-1.5 mb-1">
                          <span className="px-2 py-0.5 rounded text-[9px] font-bold uppercase bg-slate-100 dark:bg-navy-950 text-slate-700 dark:text-slate-300 border dark:border-navy-800 border-slate-200">
                            {rec.chart_type}
                          </span>
                          <span className="text-[10px] text-muted font-medium">
                            {rec.domain_category || rec.task_type}
                          </span>
                        </div>
                        <h4 className="font-semibold text-xs dark:text-coolwhite text-slate-900 group-hover:text-teal-400 transition-colors">
                          {rec.title}
                        </h4>
                      </div>

                      {/* Recommendation Score Badge */}
                      <div className="text-right shrink-0">
                        <div
                          className={`px-2 py-0.5 rounded font-mono text-[10px] font-bold ${
                            rec.score_source === 'mlvis_model'
                              ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                              : isNeural
                              ? 'bg-teal-500/15 text-teal-400 border border-teal-500/30'
                              : 'bg-blue-500/15 text-blue-400 border border-blue-500/30'
                          }`}
                        >
                          {(rec.recommendation_score * 100).toFixed(0)}%
                        </div>
                        <div className="text-[8px] text-muted mt-0.5 font-medium">
                          {rec.score_source?.startsWith('mlvis')
                            ? (rec.is_cold_start ? 'MLVis Base' : (rec.preference_applied ? 'MLVis + Boost' : 'MLVis Base'))
                            : isNeural
                            ? 'Neural Score'
                            : 'Suitability'}
                        </div>
                      </div>
                    </div>

                    {/* Warm-Start Personalization Indicator */}
                    {rec.preference_applied && rec.personalization_label && (
                      <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-semibold bg-purple-500/10 text-purple-600 dark:text-purple-400 border border-purple-500/20">
                        <Sparkles className="w-3 h-3 text-purple-500 shrink-0" />
                        <span>{rec.personalization_label}</span>
                        <span className="font-mono">
                          {rec.history_personalization_boost > 0 ? `+${(rec.history_personalization_boost * 100).toFixed(0)}%` : `${(rec.history_personalization_boost * 100).toFixed(0)}%`}
                        </span>
                        <span className="text-muted text-[9px]">
                          (Base: {((rec.base_model_confidence || 0) * 100).toFixed(0)}%)
                        </span>
                      </div>
                    )}

                    <p className="text-[11px] text-muted leading-relaxed">
                      {rec.explanation}
                    </p>


                    {/* Card Actions: Apply Chart & Feedback */}
                    <div className="pt-2 border-t dark:border-navy-800/80 border-slate-100 flex items-center justify-between">
                      <button
                        onClick={() => applyRecommendation(rec)}
                        className="flex items-center gap-1 text-[11px] font-semibold text-teal-500 hover:text-teal-400 transition-colors"
                      >
                        <span>Render This Chart</span>
                        <ChevronRight className="w-3.5 h-3.5" />
                      </button>

                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => submitFeedback(rec, 'helpful')}
                          className={`p-1 rounded text-muted hover:text-emerald-500 ${
                            fb === 'helpful' ? 'text-emerald-500 bg-emerald-500/10' : ''
                          }`}
                          title="Helpful suggestion"
                        >
                          <ThumbsUp className="w-3 h-3" />
                        </button>
                        <button
                          onClick={() => submitFeedback(rec, 'unhelpful')}
                          className={`p-1 rounded text-muted hover:text-rose-500 ${
                            fb === 'unhelpful' ? 'text-rose-500 bg-rose-500/10' : ''
                          }`}
                          title="Not helpful"
                        >
                          <ThumbsDown className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: DATASET PROFILING & SCHEMA */}
      {activeTab === 'dataset' && profile && (
        <div className="space-y-6">
          {/* Column Summary Table */}
          <div className="rounded-2xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200 overflow-hidden shadow-sm">
            <div className="p-4 border-b dark:border-navy-800 border-slate-200 flex items-center justify-between">
              <div>
                <h3 className="font-bold text-sm dark:text-coolwhite text-slate-900">
                  Dataset Schema & Statistical Properties
                </h3>
                <p className="text-xs text-muted mt-0.5">
                  Detailed inspection of column data types, unique values, distributions, and null ratios.
                </p>
              </div>
              <span className="text-xs font-mono px-2.5 py-1 rounded bg-teal-500/10 text-teal-400 border border-teal-500/30">
                {profile.columns.length} Total Columns
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="bg-slate-50 dark:bg-navy-950 text-muted uppercase font-bold text-[10px] border-b dark:border-navy-800 border-slate-200">
                  <tr>
                    <th className="px-4 py-3">Column Name</th>
                    <th className="px-4 py-3">Inferred Role</th>
                    <th className="px-4 py-3">Unique Values</th>
                    <th className="px-4 py-3">Missingness</th>
                    <th className="px-4 py-3">Sample Values</th>
                    <th className="px-4 py-3">Distribution Statistics</th>
                  </tr>
                </thead>
                <tbody className="divide-y dark:divide-navy-800 divide-slate-100 font-medium">
                  {profile.columns.map((c) => (
                    <tr key={c.name} className="hover:bg-slate-50 dark:hover:bg-navy-900/40">
                      <td className="px-4 py-3 font-semibold dark:text-coolwhite text-slate-900">
                        {c.name}
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                            c.dtype === 'numeric'
                              ? 'bg-blue-500/15 text-blue-400 border border-blue-500/30'
                              : c.dtype === 'categorical'
                              ? 'bg-purple-500/15 text-purple-400 border border-purple-500/30'
                              : 'bg-slate-500/15 text-slate-400 border border-slate-500/30'
                          }`}
                        >
                          {c.dtype}
                        </span>
                      </td>
                      <td className="px-4 py-3 font-mono">{c.unique_count.toLocaleString()}</td>
                      <td className="px-4 py-3 font-mono">
                        {c.null_count === 0 ? (
                          <span className="text-emerald-500">0 (0.0%)</span>
                        ) : (
                          <span className="text-amber-500">{c.null_count} ({c.null_pct}%)</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-muted font-mono truncate max-w-[200px]">
                        {c.sample_values?.slice(0, 3).join(', ')}
                      </td>
                      <td className="px-4 py-3 text-muted">
                        {c.dtype === 'numeric' && c.stats?.mean !== undefined ? (
                          <span className="font-mono text-[11px]">
                            mean: {c.stats.mean} · range: [{c.stats.min}, {c.stats.max}] · skew: {c.stats.skewness}
                          </span>
                        ) : c.stats?.top_categories ? (
                          <span className="text-[11px]">
                            top: {c.stats.top_categories.slice(0, 2).map((t) => `${t.category} (${t.pct}%)`).join(', ')}
                          </span>
                        ) : (
                          '-'
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Sample Rows Preview */}
          <div className="rounded-2xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200 overflow-hidden shadow-sm p-4">
            <h4 className="font-bold text-xs uppercase tracking-wider dark:text-coolwhite text-slate-900 mb-3">
              Dataset Records Preview (First 10 Rows)
            </h4>
            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono">
                <thead className="bg-slate-50 dark:bg-navy-950 text-muted uppercase text-[10px]">
                  <tr>
                    {profile.columns.slice(0, 8).map((c) => (
                      <th key={c.name} className="px-3 py-2 text-left">{c.name}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y dark:divide-navy-800 divide-slate-100">
                  {profile.sample_records?.map((r, i) => (
                    <tr key={i} className="hover:bg-slate-50 dark:hover:bg-navy-900/40">
                      {profile.columns.slice(0, 8).map((c) => (
                        <td key={c.name} className="px-3 py-2 text-muted truncate max-w-[150px]">
                          {String(r[c.name] ?? '')}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: MODEL ANALYTICS & INTERACTION METRICS */}
      {activeTab === 'analytics' && (
        <div className="space-y-6">
          {/* Measured Data Callout */}
          <div className="p-4 rounded-xl bg-teal-500/10 border border-teal-500/30 flex items-start gap-3">
            <Info className="w-5 h-5 text-teal-500 shrink-0 mt-0.5" />
            <div className="text-xs">
              <span className="font-bold text-teal-400">Live Measured Production Data: </span>
              Metrics below are computed from real interactions (impressions, clicks, feedback) persisted in the SQLite platform database. Offline benchmark evaluation numbers are displayed side-by-side for reference.
            </div>
          </div>

          {/* KPI Metrics Strip */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200">
              <div className="text-[11px] font-semibold text-muted uppercase">Total Impressions</div>
              <div className="text-2xl font-bold dark:text-coolwhite text-slate-900 mt-1">
                {analytics?.summary?.impressions?.toLocaleString() ?? 0}
              </div>
              <div className="text-[10px] text-muted mt-0.5">Recommendations shown</div>
            </div>

            <div className="p-4 rounded-xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200">
              <div className="text-[11px] font-semibold text-muted uppercase">Clicks & CTR</div>
              <div className="text-2xl font-bold dark:text-coolwhite text-slate-900 mt-1">
                {((analytics?.summary?.click_through_rate ?? 0) * 100).toFixed(1)}%
              </div>
              <div className="text-[10px] text-muted mt-0.5">
                {analytics?.summary?.clicks ?? 0} total clicks
              </div>
            </div>

            <div className="p-4 rounded-xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200">
              <div className="text-[11px] font-semibold text-muted uppercase">Helpful Feedback</div>
              <div className="text-2xl font-bold dark:text-coolwhite text-slate-900 mt-1">
                {((analytics?.summary?.helpful_ratio ?? 0) * 100).toFixed(0)}%
              </div>
              <div className="text-[10px] text-muted mt-0.5">
                {analytics?.summary?.feedback_helpful ?? 0} positive ratings
              </div>
            </div>

            <div className="p-4 rounded-xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200">
              <div className="text-[11px] font-semibold text-muted uppercase">Offline Benchmark NDCG@5</div>
              <div className="text-2xl font-bold text-teal-400 mt-1">
                {analytics?.offline_benchmark_reference?.overall_ndcg_at_5?.toFixed(4) ?? '0.1450'}
              </div>
              <div className="text-[10px] text-muted mt-0.5">
                +{analytics?.offline_benchmark_reference?.cold_start_mitigation_lift_pct ?? 28.5}% cold-start lift
              </div>
            </div>
          </div>

          {/* Charts Row: Chart Type Distribution & Cohort Performance */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Chart Type Distribution */}
            <div className="p-5 rounded-2xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200 shadow-sm space-y-3">
              <h4 className="font-bold text-xs uppercase tracking-wider dark:text-coolwhite text-slate-900">
                Chart Type Interaction Distribution
              </h4>
              <p className="text-[11px] text-muted">
                Frequency of recommendations rendered across visual modalities.
              </p>
              <div className="h-[240px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={analytics?.chart_type_distribution || []}>
                    <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1e293b' : '#f1f5f9'} />
                    <XAxis dataKey="chart_type" stroke={isDark ? '#64748b' : '#94a3b8'} fontSize={11} />
                    <YAxis stroke={isDark ? '#64748b' : '#94a3b8'} fontSize={11} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: isDark ? '#0f172a' : '#ffffff',
                        borderColor: isDark ? '#334155' : '#cbd5e1',
                        borderRadius: '8px',
                        fontSize: '12px',
                      }}
                    />
                    <Bar dataKey="count" fill="#14b8a6" radius={[4, 4, 0, 0]}>
                      {(analytics?.chart_type_distribution || []).map((_, i) => (
                        <Cell key={`dist-cell-${i}`} fill={colors[i % colors.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Cold Start vs Returning User Cohort Performance */}
            <div className="p-5 rounded-2xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200 shadow-sm space-y-4">
              <h4 className="font-bold text-xs uppercase tracking-wider dark:text-coolwhite text-slate-900">
                Cold-Start vs Returning-User Performance
              </h4>
              <p className="text-[11px] text-muted">
                Empirical click-through rate comparison across interaction history depth.
              </p>

              <div className="grid grid-cols-2 gap-4 pt-2">
                <div className="p-4 rounded-xl dark:bg-navy-950/80 bg-slate-50 border dark:border-navy-800 border-slate-200 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-xs text-blue-400">Cold-Start Sessions</span>
                    <span className="text-[10px] text-muted">&lt; 3 events</span>
                  </div>
                  <div className="text-2xl font-bold dark:text-coolwhite text-slate-900">
                    {((analytics?.cohort_performance?.cold_start?.click_through_rate ?? 0) * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-muted">
                    {analytics?.cohort_performance?.cold_start?.clicks ?? 0} clicks / {analytics?.cohort_performance?.cold_start?.impressions ?? 0} impressions
                  </div>
                </div>

                <div className="p-4 rounded-xl dark:bg-navy-950/80 bg-slate-50 border dark:border-navy-800 border-slate-200 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-xs text-emerald-400">Warm / Returning</span>
                    <span className="text-[10px] text-muted">&ge; 3 events</span>
                  </div>
                  <div className="text-2xl font-bold dark:text-coolwhite text-slate-900">
                    {((analytics?.cohort_performance?.warm_start?.click_through_rate ?? 0) * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-muted">
                    {analytics?.cohort_performance?.warm_start?.clicks ?? 0} clicks / {analytics?.cohort_performance?.warm_start?.impressions ?? 0} impressions
                  </div>
                </div>
              </div>

              <div className="text-[11px] text-muted pt-2 border-t dark:border-navy-800 border-slate-100">
                The cold-start adaptive gate mitigates performance degradation on sparse-history sessions by up-weighting dataset attributes and context signals.
              </div>
            </div>
          </div>

          {/* MLVis Model Benchmark Evaluation Card */}
          {analytics?.mlvis_benchmark_reference && (
            <div className="p-5 rounded-2xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-bold text-xs uppercase tracking-wider dark:text-coolwhite text-slate-900 flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-emerald-400" />
                    MLVis Offline Evaluation: Neural Recommender vs Statistical Baseline
                  </h4>
                  <p className="text-[11px] text-muted mt-0.5">
                    Evaluated on 783 held-out test visualizations from 140 unseen datasets (grouped splits to prevent dataset leakage).
                  </p>
                </div>
                <span className="px-2.5 py-1 rounded-full text-[10px] font-bold uppercase bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                  {analytics.mlvis_benchmark_reference.dataset || 'MLVis 1K Benchmark'}
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 rounded-xl dark:bg-navy-950/80 bg-slate-50 border dark:border-navy-800 border-slate-200">
                  <div className="text-[10px] font-bold text-muted uppercase">Top-1 Accuracy</div>
                  <div className="flex items-baseline gap-2 mt-1">
                    <span className="text-2xl font-bold text-emerald-400">
                      {((analytics.mlvis_benchmark_reference.mlvis_neural_model?.accuracy ?? 0.9068) * 100).toFixed(1)}%
                    </span>
                    <span className="text-xs text-muted">
                      vs {((analytics.mlvis_benchmark_reference.statistical_baseline?.accuracy ?? 0.3972) * 100).toFixed(1)}% baseline
                    </span>
                  </div>
                  <div className="text-[10px] text-emerald-500 font-semibold mt-1">
                    +50.9% improvement over baseline
                  </div>
                </div>

                <div className="p-4 rounded-xl dark:bg-navy-950/80 bg-slate-50 border dark:border-navy-800 border-slate-200">
                  <div className="text-[10px] font-bold text-muted uppercase">Macro F1-Score</div>
                  <div className="flex items-baseline gap-2 mt-1">
                    <span className="text-2xl font-bold text-teal-400">
                      {(analytics.mlvis_benchmark_reference.mlvis_neural_model?.macro_f1 ?? 0.7722).toFixed(4)}
                    </span>
                    <span className="text-xs text-muted">
                      vs {(analytics.mlvis_benchmark_reference.statistical_baseline?.macro_f1 ?? 0.1824).toFixed(4)} baseline
                    </span>
                  </div>
                  <div className="text-[10px] text-teal-500 font-semibold mt-1">
                    +0.5897 macro F1 lift across chart classes
                  </div>
                </div>

                <div className="p-4 rounded-xl dark:bg-navy-950/80 bg-slate-50 border dark:border-navy-800 border-slate-200">
                  <div className="text-[10px] font-bold text-muted uppercase">Ranking Quality (NDCG@3)</div>
                  <div className="flex items-baseline gap-2 mt-1">
                    <span className="text-2xl font-bold text-purple-400">
                      {(analytics.mlvis_benchmark_reference.mlvis_neural_model?.ndcg_at_3 ?? 0.9614).toFixed(4)}
                    </span>
                    <span className="text-xs text-muted">
                      vs {(analytics.mlvis_benchmark_reference.statistical_baseline?.ndcg_at_3 ?? 0.5169).toFixed(4)} baseline
                    </span>
                  </div>
                  <div className="text-[10px] text-purple-500 font-semibold mt-1">
                    +0.4446 NDCG@3 ranking gain
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Audit Trail: Recent Visual Interactions */}
          <div className="rounded-2xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200 overflow-hidden shadow-sm p-4">
            <h4 className="font-bold text-xs uppercase tracking-wider dark:text-coolwhite text-slate-900 mb-3">
              Recent Visual Interaction Event Audit Log ({interactions.length} events recorded)
            </h4>
            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono text-left">
                <thead className="bg-slate-50 dark:bg-navy-950 text-muted uppercase text-[10px]">
                  <tr>
                    <th className="px-3 py-2">Timestamp</th>
                    <th className="px-3 py-2">Event Type</th>
                    <th className="px-3 py-2">Chart ID</th>
                    <th className="px-3 py-2">Modality</th>
                    <th className="px-3 py-2">Session ID</th>
                  </tr>
                </thead>
                <tbody className="divide-y dark:divide-navy-800 divide-slate-100">
                  {interactions.slice(0, 10).map((ev) => (
                    <tr key={ev.id} className="hover:bg-slate-50 dark:hover:bg-navy-900/40">
                      <td className="px-3 py-2 text-muted">{ev.created_at ? new Date(ev.created_at).toLocaleTimeString() : '-'}</td>
                      <td className="px-3 py-2 font-bold text-teal-400">{ev.event_type}</td>
                      <td className="px-3 py-2">{ev.chart_id || '-'}</td>
                      <td className="px-3 py-2 capitalize">{ev.chart_type || '-'}</td>
                      <td className="px-3 py-2 text-muted truncate max-w-[120px]">{ev.session_id}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}


      {/* TAB 4: YOUR HISTORY & RECENTLY RECOMMENDED */}
      {activeTab === 'history' && (
        <div className="space-y-4">
          <div className="p-5 rounded-2xl dark:bg-navy-900/60 bg-white border dark:border-navy-800 border-slate-200 shadow-sm space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b dark:border-navy-800 border-slate-200 pb-4">
              <div>
                <h3 className="text-base font-bold dark:text-coolwhite text-slate-900 flex items-center gap-2">
                  <Compass className="w-5 h-5 text-teal-500" />
                  Your History & Recently Recommended
                </h3>
                <p className="text-xs text-muted mt-0.5">
                  Chronological records of visual analytics recommendations, selected charts, and explicit feedback for user <span className="font-mono font-bold text-teal-600 dark:text-teal-400">{selectedUserId}</span>.
                </p>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={fetchAnalytics}
                  className="px-3 py-1.5 rounded-xl border dark:border-navy-800 border-slate-200 text-xs font-semibold text-muted hover:dark:text-coolwhite hover:text-slate-900 transition-all flex items-center gap-1.5"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Refresh History</span>
                </button>
              </div>
            </div>

            {/* Persistent User History Component in Expanded View */}
            <UserIdentityStep
              clientId={activeClientId}
              userId={selectedUserId}
              onUserIdChange={(newUid) => setSelectedUserId(newUid)}
              showHistoryDrawer={true}
            />

            {/* Revisit Recently Explored Datasets & Charts */}
            <div className="pt-2">
              <h4 className="text-xs font-bold uppercase tracking-wider dark:text-coolwhite text-slate-800 mb-3">
                Recently Interacted Charts in This Session
              </h4>
              {interactions.filter(e => e.chart_id).length === 0 ? (
                <div className="p-8 text-center rounded-2xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 text-xs text-muted">
                  No chart selections recorded yet in this session. Click <strong>"Render This Chart"</strong> or leave feedback on recommendations to build your profile.
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {interactions.filter(e => e.chart_id).slice(0, 6).map((item, idx) => (
                    <div
                      key={item.id || idx}
                      className="p-3 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 space-y-2 hover:border-teal-500/40 transition-all"
                    >
                      <div className="flex items-center justify-between">
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-teal-500/10 text-teal-600 dark:text-teal-400">
                          {item.chart_type || 'Chart'}
                        </span>
                        <span className="text-[10px] text-muted">
                          {item.created_at ? new Date(item.created_at).toLocaleTimeString() : 'Recent'}
                        </span>
                      </div>
                      <div className="font-semibold text-xs dark:text-coolwhite text-slate-900 truncate">
                        {item.chart_id}
                      </div>
                      <div className="text-[11px] text-muted capitalize">
                        Event: <span className="text-teal-600 dark:text-teal-400 font-medium">{item.event_type.replace('_', ' ')}</span>
                      </div>
                      <div className="pt-1">
                        <button
                          onClick={() => {
                            if (item.chart_type) setChartType(item.chart_type);
                            setActiveTab('explorer');
                          }}
                          className="w-full py-1 text-center text-[11px] font-bold text-teal-600 dark:text-teal-400 hover:underline bg-teal-500/10 rounded-lg cursor-pointer"
                        >
                          Revisit In Explorer →
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

