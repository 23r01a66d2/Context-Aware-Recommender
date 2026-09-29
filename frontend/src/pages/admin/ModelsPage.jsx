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
  const [loading, setLoading] = useState(true);
  const [activating, setActivating] = useState(null);
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  const fetchModels = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getModels(activeId);
      setModels(data);
    } catch (err) {
      setError(err.message || 'Failed to load model versions.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, [activeId]);

  const handleActivate = async (versionTag) => {
    setActivating(versionTag);
    setError(null);
    setSuccessMsg(null);
    try {
      await api.activateModel(activeId, versionTag);
      setSuccessMsg(`Model version '${versionTag}' is now the active inference model.`);
      await fetchModels();
      await refreshClients();
    } catch (err) {
      setError(err.message || `Failed to activate model version ${versionTag}`);
    } finally {
      setActivating(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
            <Boxes className="w-6 h-6 text-teal-400" />
            Model Registry & Version Management
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Client: <span className="font-mono text-emerald-400 font-semibold">{activeId}</span>
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={fetchModels}
            disabled={loading}
            className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
            title="Refresh Models"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <Link
            to={`/admin/clients/${activeId}/analytics`}
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold text-xs shadow-sm transition-all"
          >
            <BarChart3 className="w-3.5 h-3.5" />
            Inspect Offline Analytics
          </Link>
        </div>
      </div>

      {/* Messages */}
      {error && (
        <div className="p-4 rounded-lg bg-rose-950/70 border border-rose-800/80 text-rose-300 text-xs flex items-start gap-2.5">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-400" />
          <div>{error}</div>
        </div>
      )}
      {successMsg && (
        <div className="p-4 rounded-lg bg-emerald-950/70 border border-emerald-800/80 text-emerald-300 text-xs flex items-start gap-2.5">
          <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5 text-emerald-400" />
          <div>{successMsg}</div>
        </div>
      )}

      {/* Model Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {loading ? (
          <div className="col-span-2 py-12 text-center text-slate-500 text-xs">
            Loading registered models...
          </div>
        ) : models.length === 0 ? (
          <div className="col-span-2 py-12 text-center bg-slate-900/40 border border-dashed border-slate-800 rounded-xl text-slate-400 text-xs space-y-3">
            <div>No model versions registered for client '{activeId}' yet.</div>
            <Link
              to={`/admin/clients/${activeId}/training`}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs"
            >
              <Play className="w-3.5 h-3.5" /> Start Training Run
            </Link>
          </div>
        ) : (
          models.map((m) => {
            const isActive = m.is_active;
            const dims = m.feature_dimensions || {};

            return (
              <div
                key={m.version_tag}
                className={`bg-slate-900/90 border rounded-xl p-5 shadow-sm space-y-4 transition-all ${
                  isActive
                    ? 'border-emerald-500/50 ring-1 ring-emerald-500/30 bg-emerald-950/[0.08]'
                    : 'border-slate-800'
                }`}
              >
                {/* Top: Version & Status */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-base font-extrabold text-white">
                      {m.version_tag}
                    </span>
                    {isActive ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                        <CheckCircle2 className="w-3 h-3" /> Active Inference Target
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 rounded-full text-[11px] bg-slate-800 text-slate-400 border border-slate-700">
                        Inactive
                      </span>
                    )}
                  </div>

                  <span className="text-[11px] text-slate-500 font-mono">
                    Run: {m.training_run_id || 'manual'}
                  </span>
                </div>

                {/* Feature Dimensions Snapshot */}
                <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80 text-xs">
                  <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-teal-400" />
                    Encoder Feature Dimensions
                  </div>
                  <div className="grid grid-cols-3 gap-2 font-mono text-[11px]">
                    <div className="bg-slate-900 p-2 rounded border border-slate-800">
                      <div className="text-slate-500">Behavior</div>
                      <div className="text-indigo-400 font-bold">{dims.behavior ?? '—'} dims</div>
                    </div>
                    <div className="bg-slate-900 p-2 rounded border border-slate-800">
                      <div className="text-slate-500">Content</div>
                      <div className="text-emerald-400 font-bold">{dims.content ?? '—'} dims</div>
                    </div>
                    <div className="bg-slate-900 p-2 rounded border border-slate-800">
                      <div className="text-slate-500">Context</div>
                      <div className="text-sky-400 font-bold">{dims.context ?? '—'} dims</div>
                    </div>
                  </div>
                </div>

                {/* Metadata & Actions */}
                <div className="flex items-center justify-between pt-2 border-t border-slate-800/80 text-xs">
                  <div className="text-[11px] text-slate-400 flex items-center gap-1">
                    <Calendar className="w-3 h-3 text-slate-500" />
                    {m.created_at ? new Date(m.created_at).toLocaleDateString() : '—'}
                  </div>

                  {!isActive && (
                    <button
                      onClick={() => handleActivate(m.version_tag)}
                      disabled={activating === m.version_tag}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-xs border border-slate-700 transition-colors"
                    >
                      {activating === m.version_tag ? (
                        <>
                          <Loader2 className="w-3 h-3 animate-spin" />
                          Activating...
                        </>
                      ) : (
                        'Activate Model'
                      )}
                    </button>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
