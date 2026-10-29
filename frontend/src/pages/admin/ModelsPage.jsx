import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useClient } from '../../context/ClientContext';
import { api } from '../../services/api';
import {
  Boxes,
  CheckCircle2,
  AlertCircle,
  Play,
  BarChart3,
  Calendar,
  Layers,
  Sparkles,
  Loader2,
  RefreshCw,
  LineChart,
  Target,
  Cpu,
  ArrowRight
} from 'lucide-react';

export default function ModelsPage() {
  const { clientId } = useParams();
  const { selectedClientId, selectClient, refreshClients } = useClient();
  const activeId = clientId || selectedClientId;

  useEffect(() => {
    if (clientId && clientId !== selectedClientId) {
      selectClient(clientId);
    }
  }, [clientId, selectedClientId, selectClient]);

  const [models, setModels] = useState([]);
  const [globalModels, setGlobalModels] = useState([]);
  const [viewScope, setViewScope] = useState('all'); // 'all' or 'client'
  const [loading, setLoading] = useState(true);
  const [activating, setActivating] = useState(null);
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  const fetchModels = async () => {
    setLoading(true);
    setError(null);
    try {
      const [clientData, globalData] = await Promise.all([
        activeId ? api.getModels(activeId).catch(() => []) : Promise.resolve([]),
        api.getAllModels().catch(() => [])
      ]);
      setModels(clientData || []);
      setGlobalModels(globalData || []);
    } catch (err) {
      setError(err.message || 'Failed to load model registry.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, [activeId]);

  const handleActivate = async (targetClientId, versionTag) => {
    setActivating(versionTag);
    setError(null);
    setSuccessMsg(null);
    try {
      await api.activateModel(targetClientId, versionTag);
      setSuccessMsg(`Model version '${versionTag}' is now the active inference model for ${targetClientId}.`);
      await fetchModels();
      await refreshClients();
    } catch (err) {
      setError(err.message || `Failed to activate model version ${versionTag}`);
    } finally {
      setActivating(null);
    }
  };

  const displayedModels = viewScope === 'all' ? globalModels : models;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold dark:text-coolwhite text-slate-900 tracking-tight flex items-center gap-2">
            <Boxes className="w-5 h-5 text-teal-500" />
            Model Registry & Intelligence Workspace
          </h1>
          <p className="text-xs dark:text-muted text-slate-500 mt-0.5">
            Discover, evaluate, and open interactive workspaces for visualization and contextual recommender models.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Scope Toggle */}
          <div className="flex items-center bg-slate-100 dark:bg-navy-900 p-1 rounded-xl border dark:border-navy-800 border-slate-200 text-xs">
            <button
              onClick={() => setViewScope('all')}
              className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                viewScope === 'all'
                  ? 'bg-white dark:bg-navy-800 text-teal-600 dark:text-teal-400 font-bold shadow-sm'
                  : 'text-muted hover:text-coolwhite'
              }`}
            >
              All Models ({globalModels.length})
            </button>
            <button
              onClick={() => setViewScope('client')}
              className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                viewScope === 'client'
                  ? 'bg-white dark:bg-navy-800 text-teal-600 dark:text-teal-400 font-bold shadow-sm'
                  : 'text-muted hover:text-coolwhite'
              }`}
            >
              Client: {activeId} ({models.length})
            </button>
          </div>

          <button
            onClick={fetchModels}
            disabled={loading}
            className="p-2 rounded-xl dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 dark:text-muted text-slate-600 hover:dark:text-coolwhite hover:text-slate-900 transition-colors cursor-pointer shadow-sm"
            title="Refresh Models"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <Link
            to={`/admin/clients/${activeId}/analytics`}
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-teal-500 hover:bg-teal-400 text-navy-950 font-bold text-xs shadow-sm transition-all"
          >
            <BarChart3 className="w-3.5 h-3.5" />
            Inspect Analytics
          </Link>
        </div>
      </div>

      {/* Messages */}
      {error && (
        <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-300 dark:border-rose-800/80 text-rose-700 dark:text-rose-300 text-xs flex items-start gap-2.5">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-500" />
          <div>{error}</div>
        </div>
      )}
      {successMsg && (
        <div className="p-4 rounded-2xl bg-teal-500/10 border border-teal-300 dark:border-teal-500/30 text-teal-700 dark:text-teal-300 text-xs flex items-start gap-2.5">
          <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5 text-teal-500" />
          <div>{successMsg}</div>
        </div>
      )}

      {/* Model Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {loading ? (
          <div className="col-span-2 py-10 text-center dark:text-muted text-slate-500 text-xs">
            Loading registered models...
          </div>
        ) : displayedModels.length === 0 ? (
          <div className="col-span-2 py-10 text-center dark:bg-navy-900 bg-white border border-dashed dark:border-navy-800 border-slate-300 rounded-2xl dark:text-muted text-slate-500 text-xs space-y-3">
            <div>No model versions registered for view scope '{viewScope}'.</div>
            <Link
              to={`/admin/clients/${activeId}/training`}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-teal-500 hover:bg-teal-400 text-navy-950 font-bold text-xs"
            >
              <Play className="w-3.5 h-3.5" /> Start Training Run
            </Link>
          </div>
        ) : (
          displayedModels.map((m) => {
            const isActive = m.is_active;
            const dims = m.feature_dimensions || {};
            const isMLVis = m.client_id === 'mlvis' || m.supports_visualizations;
            const modelKey = m.model_id || `${m.client_id}_${m.version || m.version_tag}`;
            const targetClient = m.client_id || activeId;
            const versionTag = m.version || m.version_tag || 'v1';

            return (
              <div
                key={modelKey}
                className={`dark:bg-navy-900 bg-white border rounded-2xl p-5 shadow-sm space-y-4 transition-all hover:border-teal-500/50 ${
                  isActive
                    ? 'border-teal-500 ring-1 ring-teal-500/20 dark:bg-navy-850 bg-teal-50/20'
                    : 'dark:border-navy-800 border-slate-200'
                }`}
              >
                {/* Top: Version, Name & Status */}
                <div className="flex items-start justify-between gap-3">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-base font-bold dark:text-coolwhite text-slate-900">
                        {m.name || `${m.client_name || targetClient} (${versionTag})`}
                      </span>
                      {isActive ? (
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-300 dark:border-teal-500/20">
                          <CheckCircle2 className="w-3 h-3 text-teal-500" /> Active Target
                        </span>
                      ) : (
                        <span className="px-2.5 py-0.5 rounded-full text-[11px] dark:bg-navy-950 bg-slate-100 dark:text-muted text-slate-500 border dark:border-navy-800 border-slate-200">
                          {m.training_status || 'Completed'}
                        </span>
                      )}
                    </div>
                    <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted">
                      <span>Task: <strong className="text-teal-600 dark:text-teal-400">{m.supported_task || (isMLVis ? 'Chart Recommendation' : 'Product Recommendation')}</strong></span>
                      <span>•</span>
                      <span>Version: <strong className="font-mono">{versionTag}</strong></span>
                      <span>•</span>
                      <span>Client: <strong className="font-mono">{targetClient}</strong></span>
                    </div>
                  </div>

                  {/* Open Model Workspace Button */}
                  <Link
                    to={`/admin/models/${targetClient}/workspace?tab=overview`}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-teal-500 hover:bg-teal-400 text-navy-950 font-bold text-xs transition-all shadow-sm shrink-0"
                  >
                    <span>Open Workspace</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </Link>
                </div>

                {/* Architecture and Modalities */}
                <div className="p-3 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 text-xs space-y-2">
                  <div className="text-[11px] font-semibold dark:text-muted text-slate-500 uppercase tracking-wider flex items-center justify-between">
                    <span className="flex items-center gap-1.5">
                      <Cpu className="w-3.5 h-3.5 text-teal-500" />
                      {m.model_type || (isMLVis ? 'MLVis Neural Recommender' : 'Two-Tower Adaptive Fusion')}
                    </span>
                    {isMLVis && (
                      <span className="px-2 py-0.5 rounded text-[10px] bg-teal-500/10 text-teal-600 dark:text-teal-400 font-bold">
                        Supports Visualizations
                      </span>
                    )}
                  </div>
                  <div className="text-[11px] text-muted">
                    {m.architecture || (isMLVis ? 'Deep Neural Classifier for Chart Types' : 'Two-Tower Adaptive Fusion with Cold-Start Gating')}
                  </div>
                  {dims && Object.keys(dims).length > 0 && (
                    <div className="grid grid-cols-3 gap-2 font-mono text-[11px] pt-1">
                      {Object.entries(dims).slice(0, 3).map(([k, v]) => (
                        <div key={k} className="dark:bg-navy-900 bg-white p-2 rounded-lg border dark:border-navy-800 border-slate-200">
                          <div className="dark:text-muted text-slate-500 capitalize">{k}</div>
                          <div className="text-teal-600 dark:text-teal-400 font-bold">{v} dims</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Evaluation snapshot */}
                {m.metrics && Object.keys(m.metrics).length > 0 && (
                  <div className="flex flex-wrap items-center gap-3 text-xs dark:bg-navy-950/60 bg-slate-100 p-2.5 rounded-xl border dark:border-navy-800 border-slate-200">
                    <div className="text-[10px] uppercase font-bold text-muted flex items-center gap-1">
                      <BarChart3 className="w-3 h-3 text-teal-500" />
                      Key Metrics:
                    </div>
                    {isMLVis ? (
                      <>
                        <div className="text-[11px]">Accuracy: <strong className="font-mono text-teal-600 dark:text-teal-400">{((m.metrics.accuracy || 0) * 100).toFixed(1)}%</strong></div>
                        <div className="text-[11px]">Macro F1: <strong className="font-mono text-teal-600 dark:text-teal-400">{(m.metrics.macro_f1 || 0).toFixed(3)}</strong></div>
                        <div className="text-[11px]">Precision@3: <strong className="font-mono text-teal-600 dark:text-teal-400">{((m.metrics.precision_at_3 || 0) * 100).toFixed(1)}%</strong></div>
                      </>
                    ) : (
                      <>
                        {m.metrics.ndcg_at_5 && <div className="text-[11px]">NDCG@5: <strong className="font-mono text-teal-600 dark:text-teal-400">{(m.metrics.ndcg_at_5).toFixed(3)}</strong></div>}
                        {m.metrics.hit_rate_at_5 && <div className="text-[11px]">Hit Rate@5: <strong className="font-mono text-teal-600 dark:text-teal-400">{(m.metrics.hit_rate_at_5).toFixed(3)}</strong></div>}
                        {m.metrics.test_metric && <div className="text-[11px]">Validation: <strong className="font-mono text-teal-600 dark:text-teal-400">{m.metrics.test_metric}</strong></div>}
                      </>
                    )}
                  </div>
                )}

                {/* Action Bar */}
                <div className="flex items-center justify-between pt-2 border-t dark:border-navy-800 border-slate-200 text-xs">
                  <div className="text-[11px] dark:text-muted text-slate-500 flex items-center gap-1">
                    <Calendar className="w-3 h-3 text-slate-400" />
                    {m.created_at ? new Date(m.created_at).toLocaleDateString() : '—'}
                  </div>

                  <div className="flex items-center gap-2">
                    {/* Visualizations workspace shortcut */}
                    {isMLVis ? (
                      <Link
                        to={`/admin/models/${targetClient}/workspace?tab=visualizations`}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-teal-500/10 text-teal-600 dark:text-teal-400 hover:bg-teal-500/20 font-semibold text-xs border border-teal-500/30 transition-colors"
                      >
                        <LineChart className="w-3 h-3" />
                        Visualizations
                      </Link>
                    ) : (
                      <Link
                        to={`/admin/models/${targetClient}/workspace?tab=visualizations`}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-navy-800 text-muted hover:text-coolwhite font-medium text-xs border dark:border-navy-700 border-slate-200 transition-colors"
                      >
                        <LineChart className="w-3 h-3" />
                        Visualizations
                      </Link>
                    )}

                    {!isActive && targetClient !== 'mlvis' && (
                      <button
                        onClick={() => handleActivate(targetClient, versionTag)}
                        disabled={activating === versionTag}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg dark:bg-navy-800 bg-slate-100 hover:dark:bg-navy-700 hover:bg-slate-200 dark:text-coolwhite text-slate-800 font-semibold text-xs border dark:border-navy-700 border-slate-200 transition-colors cursor-pointer"
                      >
                        {activating === versionTag ? (
                          <>
                            <Loader2 className="w-3 h-3 animate-spin" />
                            Activating...
                          </>
                        ) : (
                          'Activate'
                        )}
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
