import React, { useState, useEffect } from 'react';
import { useParams, useSearchParams, Link, useNavigate } from 'react-router-dom';
import { useClient } from '../../context/ClientContext';
import { api } from '../../services/api';
import VisualAnalyticsPage from './VisualAnalyticsPage';
import {
  Boxes,
  LineChart,
  BarChart3,
  Sparkles,
  ArrowLeft,
  CheckCircle2,
  AlertCircle,
  Play,
  Calendar,
  Layers,
  Cpu,
  RefreshCw,
  ExternalLink,
  Target,
  FileSpreadsheet,
  Activity,
  Award,
  Info
} from 'lucide-react';

export default function ModelWorkspacePage() {
  const { modelId } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const { selectedClientId } = useClient();

  const currentTab = searchParams.get('tab') || 'overview';
  const setTab = (tab) => {
    setSearchParams({ tab });
  };

  const [allModels, setAllModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadModels() {
      setLoading(true);
      setError(null);
      try {
        const data = await api.getAllModels();
        setAllModels(data);
      } catch (err) {
        setError(err.message || 'Failed to load model registry.');
      } finally {
        setLoading(false);
      }
    }
    loadModels();
  }, []);

  // Find targeted model
  const activeModel = React.useMemo(() => {
    if (!allModels || allModels.length === 0) return null;
    const cleanTarget = (modelId || '').toLowerCase();
    
    // Exact or partial match on model_id, client_id, or name
    return (
      allModels.find(
        (m) =>
          m.model_id?.toLowerCase() === cleanTarget ||
          m.client_id?.toLowerCase() === cleanTarget ||
          (cleanTarget === 'mlvis' && m.client_id === 'mlvis') ||
          m.name?.toLowerCase().includes(cleanTarget)
      ) ||
      allModels[0]
    );
  }, [allModels, modelId]);

  const isMLVis = activeModel?.client_id === 'mlvis' || activeModel?.supports_visualizations;
  const metrics = activeModel?.metrics || {};
  const evalReport = activeModel?.eval_report || {};
  const dims = activeModel?.feature_dimensions || {};

  return (
    <div className="space-y-6 pb-12">
      {/* Top Breadcrumb & Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b dark:border-navy-800 border-slate-200 pb-4">
        <div className="flex items-center gap-3">
          <Link
            to="/admin/models"
            className="p-2 rounded-xl border dark:border-navy-800 border-slate-200 hover:dark:bg-navy-900 hover:bg-slate-100 transition-colors text-muted hover:text-coolwhite"
            title="Back to Models"
          >
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20">
                Model Workspace
              </span>
              <span className="text-xs text-muted font-mono">{activeModel?.client_id}</span>
              {activeModel?.is_active && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                  <CheckCircle2 className="w-3 h-3 text-emerald-500" /> Active Serving
                </span>
              )}
            </div>
            <h1 className="text-xl sm:text-2xl font-bold dark:text-coolwhite text-slate-900 tracking-tight flex items-center gap-2 mt-1">
              {isMLVis ? (
                <Sparkles className="w-6 h-6 text-teal-500" />
              ) : (
                <Boxes className="w-6 h-6 text-teal-500" />
              )}
              {activeModel?.name || (loading ? 'Loading Model...' : 'Model Details')}
            </h1>
          </div>
        </div>

        {/* Quick model switcher dropdown */}
        <div className="flex items-center gap-2 bg-slate-100 dark:bg-navy-900 border dark:border-navy-800 border-slate-200 px-3 py-1.5 rounded-xl">
          <Boxes className="w-4 h-4 text-teal-500" />
          <div className="text-left">
            <div className="text-[10px] uppercase font-bold text-muted">Switch Model</div>
            <select
              aria-label="Switch Model"
              value={activeModel?.client_id || ''}
              onChange={(e) => {
                const target = e.target.value;
                navigate(`/admin/models/${target}/workspace?tab=${currentTab}`);
              }}
              className="bg-transparent font-medium text-xs dark:text-coolwhite text-slate-800 focus:outline-none cursor-pointer"
            >
              {allModels.map((m) => (
                <option key={m.model_id || m.client_id} value={m.client_id} className="dark:bg-navy-900 bg-white">
                  {m.name || m.client_name} ({m.version || m.version_tag})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-2 border-b dark:border-navy-800 border-slate-200 pb-2">
        <button
          onClick={() => setTab('overview')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
            currentTab === 'overview'
              ? 'bg-teal-500/15 text-teal-600 dark:text-teal-300 border border-teal-500/30 shadow-sm'
              : 'text-muted hover:text-coolwhite hover:bg-navy-900/40'
          }`}
        >
          <Info className="w-4 h-4" />
          <span>Overview & Architecture</span>
        </button>

        <button
          onClick={() => setTab('visualizations')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
            currentTab === 'visualizations'
              ? 'bg-teal-500/15 text-teal-600 dark:text-teal-300 border border-teal-500/30 shadow-sm'
              : 'text-muted hover:text-coolwhite hover:bg-navy-900/40'
          }`}
        >
          <LineChart className="w-4 h-4" />
          <span>Visualizations & Analytics</span>
          {isMLVis && (
            <span className="px-1.5 py-0.2 rounded-full text-[9px] bg-teal-500 text-white font-bold uppercase">
              MLVis
            </span>
          )}
        </button>

        <button
          onClick={() => setTab('evaluation')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
            currentTab === 'evaluation'
              ? 'bg-teal-500/15 text-teal-600 dark:text-teal-300 border border-teal-500/30 shadow-sm'
              : 'text-muted hover:text-coolwhite hover:bg-navy-900/40'
          }`}
        >
          <BarChart3 className="w-4 h-4" />
          <span>Evaluation & Benchmark</span>
        </button>
      </div>

      {/* Tab Contents */}
      {currentTab === 'overview' && (
        <div className="space-y-6">
          {/* Metadata Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="p-4 rounded-2xl dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200">
              <div className="text-[11px] uppercase font-bold text-muted flex items-center gap-1.5 mb-1">
                <Target className="w-3.5 h-3.5 text-teal-500" />
                Supported Task
              </div>
              <div className="text-sm font-bold dark:text-coolwhite text-slate-800">
                {activeModel?.supported_task || 'Catalog Recommendation'}
              </div>
            </div>

            <div className="p-4 rounded-2xl dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200">
              <div className="text-[11px] uppercase font-bold text-muted flex items-center gap-1.5 mb-1">
                <Cpu className="w-3.5 h-3.5 text-teal-500" />
                Model Type
              </div>
              <div className="text-sm font-bold dark:text-coolwhite text-slate-800">
                {activeModel?.model_type || 'Neural Recommender'}
              </div>
            </div>

            <div className="p-4 rounded-2xl dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200">
              <div className="text-[11px] uppercase font-bold text-muted flex items-center gap-1.5 mb-1">
                <Activity className="w-3.5 h-3.5 text-teal-500" />
                Training Status
              </div>
              <div className="text-sm font-bold text-emerald-600 dark:text-emerald-400">
                {activeModel?.training_status || 'Production Ready'}
              </div>
            </div>

            <div className="p-4 rounded-2xl dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200">
              <div className="text-[11px] uppercase font-bold text-muted flex items-center gap-1.5 mb-1">
                <Layers className="w-3.5 h-3.5 text-teal-500" />
                Version Tag
              </div>
              <div className="text-sm font-bold font-mono dark:text-coolwhite text-slate-800">
                {activeModel?.version || activeModel?.version_tag || '1.0.0'}
              </div>
            </div>
          </div>

          {/* Architecture & Modalities */}
          <div className="p-6 rounded-2xl dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 space-y-4">
            <h3 className="text-base font-bold dark:text-coolwhite text-slate-900 flex items-center gap-2">
              <Layers className="w-4 h-4 text-teal-500" />
              Architecture & Input Features
            </h3>
            <p className="text-xs text-muted leading-relaxed">
              {activeModel?.architecture || 'Deep neural multi-modal representation architecture.'}
            </p>

            <div className="space-y-2 pt-2">
              <div className="text-xs font-semibold text-muted uppercase tracking-wider">
                Supported Modalities & Channels
              </div>
              <div className="flex flex-wrap gap-2">
                {(activeModel?.modalities || ['Features']).map((mod, i) => (
                  <span
                    key={i}
                    className="px-3 py-1 rounded-xl text-xs font-medium dark:bg-navy-950 bg-slate-100 border dark:border-navy-800 border-slate-200 dark:text-coolwhite text-slate-800"
                  >
                    {mod}
                  </span>
                ))}
              </div>
            </div>

            {dims && Object.keys(dims).length > 0 && (
              <div className="space-y-2 pt-2">
                <div className="text-xs font-semibold text-muted uppercase tracking-wider">
                  Feature Dimensions Breakdown
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                  {Object.entries(dims).map(([k, v]) => (
                    <div
                      key={k}
                      className="p-3 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 text-center"
                    >
                      <div className="text-[11px] text-muted capitalize">{k.replace('_', ' ')}</div>
                      <div className="text-sm font-bold font-mono text-teal-500 mt-0.5">{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {currentTab === 'visualizations' && (
        <VisualAnalyticsPage
          embeddedInModelWorkspace={true}
          lockedModel={isMLVis ? 'mlvis' : null}
          modelName={activeModel?.name}
          supportsVisualizations={isMLVis}
        />
      )}

      {currentTab === 'evaluation' && (
        <div className="space-y-6">
          <div className="p-6 rounded-2xl dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 space-y-4">
            <h3 className="text-base font-bold dark:text-coolwhite text-slate-900 flex items-center gap-2">
              <Award className="w-4 h-4 text-teal-500" />
              Evaluation Metrics & Performance
            </h3>

            {isMLVis && evalReport.mlvis_neural_model ? (
              <div className="space-y-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="p-4 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200">
                    <div className="text-[11px] uppercase font-bold text-muted">Test Accuracy</div>
                    <div className="text-xl font-bold font-mono text-teal-500 mt-1">
                      {((evalReport.mlvis_neural_model.accuracy || 0) * 100).toFixed(1)}%
                    </div>
                    <div className="text-[10px] text-muted mt-0.5">
                      Baseline: {((evalReport.statistical_baseline?.accuracy || 0) * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div className="p-4 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200">
                    <div className="text-[11px] uppercase font-bold text-muted">Macro F1 Score</div>
                    <div className="text-xl font-bold font-mono text-teal-500 mt-1">
                      {(evalReport.mlvis_neural_model.macro_f1 || 0).toFixed(3)}
                    </div>
                    <div className="text-[10px] text-muted mt-0.5">
                      Baseline: {(evalReport.statistical_baseline?.macro_f1 || 0).toFixed(3)}
                    </div>
                  </div>

                  <div className="p-4 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200">
                    <div className="text-[11px] uppercase font-bold text-muted">Precision@3</div>
                    <div className="text-xl font-bold font-mono text-teal-500 mt-1">
                      {((evalReport.mlvis_neural_model.precision_at_3 || 0) * 100).toFixed(1)}%
                    </div>
                    <div className="text-[10px] text-muted mt-0.5">Top-3 Chart Candidate Match</div>
                  </div>

                  <div className="p-4 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200">
                    <div className="text-[11px] uppercase font-bold text-muted">NDCG@3</div>
                    <div className="text-xl font-bold font-mono text-teal-500 mt-1">
                      {(evalReport.mlvis_neural_model.ndcg_at_3 || 0).toFixed(3)}
                    </div>
                    <div className="text-[10px] text-muted mt-0.5">Ranking Quality</div>
                  </div>
                </div>

                <div className="p-4 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 text-xs space-y-2">
                  <div className="font-semibold dark:text-coolwhite text-slate-800">
                    MLVis Benchmark Attribution:
                  </div>
                  <p className="text-muted text-[11px] leading-relaxed">
                    Evaluated on 783 unseen test visualization configurations from the official kdd21-MLVis 1K benchmark sample across 6 chart families (bar, line, scatter, box, histogram, heatmap).
                  </p>
                </div>
              </div>
            ) : metrics && Object.keys(metrics).length > 0 ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(metrics).map(([k, v]) => (
                  <div
                    key={k}
                    className="p-4 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200"
                  >
                    <div className="text-[11px] uppercase font-bold text-muted">{k.replace(/_/g, ' ')}</div>
                    <div className="text-lg font-bold font-mono text-teal-500 mt-1">
                      {typeof v === 'number' ? (v < 1 ? (v * 100).toFixed(2) + '%' : v.toFixed(2)) : String(v)}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="py-8 text-center text-muted text-xs">
                No offline evaluation records registered for this model version.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
