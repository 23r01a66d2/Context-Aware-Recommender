import React, { useState, useEffect, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useClient } from '../../context/ClientContext';
import { api } from '../../services/api';
import {
  Cpu,
  Play,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Boxes,
  BarChart3,
  TrendingDown,
  RefreshCw,
  Clock,
  Sparkles,
} from 'lucide-react';

const STAGES = ['NOT_STARTED', 'PREPROCESSING', 'TRAINING', 'EVALUATING', 'COMPLETED'];

export default function TrainingPage() {
  const { clientId } = useParams();
  const { selectedClientId, selectClient, refreshClients } = useClient();
  const activeId = clientId || selectedClientId;

  useEffect(() => {
    if (clientId && clientId !== selectedClientId) {
      selectClient(clientId);
    }
  }, [clientId, selectedClientId, selectClient]);

  const [statusData, setStatusData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [triggering, setTriggering] = useState(false);
  const [error, setError] = useState(null);

  // Hyperparameters
  const [epochs, setEpochs] = useState(15);
  const [batchSize, setBatchSize] = useState(256);
  const [learningRate, setLearningRate] = useState(0.001);

  const pollingRef = useRef(null);

  const fetchStatus = async () => {
    try {
      const data = await api.getTrainingStatus(activeId);
      setStatusData(data);
      return data;
    } catch (err) {
      console.error('Failed to fetch training status:', err);
      return null;
    }
  };

  useEffect(() => {
    setLoading(true);
    fetchStatus().finally(() => setLoading(false));
  }, [activeId]);

  // Real-time polling when training is active
  useEffect(() => {
    const isActivelyRunning =
      statusData &&
      statusData.run_id &&
      statusData.run_id !== 'none' &&
      ['NOT_STARTED', 'PREPROCESSING', 'TRAINING', 'EVALUATING'].includes(statusData.status);

    if (isActivelyRunning) {
      pollingRef.current = setInterval(async () => {
        const updated = await fetchStatus();
        if (updated && (updated.status === 'COMPLETED' || updated.status === 'FAILED')) {
          clearInterval(pollingRef.current);
          if (updated.status === 'COMPLETED' && refreshClients) {
            refreshClients();
          }
        }
      }, 2000);
    }

    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, [statusData?.status, statusData?.run_id, activeId]);

  const handleStartTraining = async (e) => {
    e.preventDefault();
    setTriggering(true);
    setError(null);

    try {
      const resp = await api.triggerTraining(activeId, {
        epochs: Number(epochs),
        batch_size: Number(batchSize),
        learning_rate: Number(learningRate),
      });
      setStatusData(resp);
    } catch (err) {
      setError(err.message || 'Failed to trigger training run.');
    } finally {
      setTriggering(false);
    }
  };

  const hasRun = Boolean(statusData?.run_id && statusData.run_id !== 'none');
  const currentStatus = hasRun ? (statusData?.status || 'NOT_STARTED') : 'IDLE';
  const isRunning =
    hasRun && ['NOT_STARTED', 'PREPROCESSING', 'TRAINING', 'EVALUATING'].includes(statusData?.status);
  const isCompleted = currentStatus === 'COMPLETED';
  const isFailed = currentStatus === 'FAILED';

  const currentEpoch = statusData?.current_epoch || 0;
  const totalEpochs = statusData?.total_epochs || epochs;
  const progressPct = totalEpochs > 0 ? Math.min(100, (currentEpoch / totalEpochs) * 100) : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
            <Cpu className="w-6 h-6 text-indigo-400" />
            Background Model Training Lifecycle
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Client: <span className="font-mono text-emerald-400 font-semibold">{activeId}</span>
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={fetchStatus}
            className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
            title="Refresh Status"
          >
            <RefreshCw className={`w-4 h-4 ${isRunning ? 'animate-spin text-indigo-400' : ''}`} />
          </button>
          {isCompleted && (
            <Link
              to={`/admin/clients/${activeId}/analytics`}
              className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold text-xs shadow-sm transition-all"
            >
              <BarChart3 className="w-3.5 h-3.5" />
              View Offline Metrics
            </Link>
          )}
        </div>
      </div>

      {/* Error alert */}
      {error && (
        <div className="p-4 rounded-lg bg-rose-950/70 border border-rose-800/80 text-rose-300 text-xs flex items-start gap-2.5">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-400" />
          <div>{error}</div>
        </div>
      )}

      {/* Pipeline Lifecycle Stages Visualization */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6">
        <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-4">
          Training Pipeline State
        </h2>

        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
          {STAGES.map((st, idx) => {
            const currentIdx = STAGES.indexOf(currentStatus);
            const isStagePassed = currentIdx >= idx;
            const isStageActive = currentStatus === st;

            return (
              <div
                key={st}
                className={`p-3 rounded-lg border text-center transition-all ${
                  isStageActive
                    ? 'bg-indigo-500/10 border-indigo-500/40 text-indigo-300 font-bold animate-pulse'
                    : isStagePassed
                    ? 'bg-slate-950/80 border-emerald-500/30 text-emerald-400'
                    : 'bg-slate-950/40 border-slate-800 text-slate-600'
                }`}
              >
                <div className="text-[10px] font-mono text-slate-500">Step {idx + 1}</div>
                <div className="text-xs mt-1 truncate">{st}</div>
              </div>
            );
          })}
        </div>

        {/* Real-time Progress Bar & Status Details */}
        {isRunning && (
          <div className="mt-6 p-4 rounded-xl bg-slate-950/80 border border-slate-800/80 space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-semibold flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
                {currentStatus === 'PREPROCESSING' && 'Preprocessing, Chronological Split & Leakage Audit...'}
                {currentStatus === 'TRAINING' && `Training Epochs (${currentEpoch} / ${totalEpochs})`}
                {currentStatus === 'EVALUATING' && 'Evaluating Precision, NDCG, Cold vs Warm metrics...'}
                {currentStatus === 'NOT_STARTED' && 'Initializing background job worker...'}
              </span>
              <span className="font-mono text-slate-400">{progressPct.toFixed(0)}%</span>
            </div>

            <div className="h-2.5 w-full bg-slate-800 rounded-full overflow-hidden">
              <div
                style={{ width: `${progressPct}%` }}
                className="h-full bg-gradient-to-r from-indigo-500 to-teal-400 transition-all duration-300"
              />
            </div>

            {statusData?.metrics && (
              <div className="flex items-center gap-6 pt-2 text-xs font-mono text-slate-300">
                {statusData.metrics.train_loss !== undefined && (
                  <div className="flex items-center gap-1.5">
                    <TrendingDown className="w-3.5 h-3.5 text-indigo-400" />
                    <span>Train Loss: {statusData.metrics.train_loss}</span>
                  </div>
                )}
                {statusData.metrics.val_loss !== undefined && (
                  <div className="flex items-center gap-1.5">
                    <TrendingDown className="w-3.5 h-3.5 text-teal-400" />
                    <span>Val Loss: {statusData.metrics.val_loss}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {isCompleted && (
          <div className="mt-6 p-4 rounded-xl bg-emerald-950/40 border border-emerald-800/60 text-emerald-300 text-xs flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              <div>
                <span className="font-bold">Training Complete!</span> Model registered into registry and active.
                {statusData?.completed_at && (
                  <div className="text-[11px] text-slate-400 font-mono mt-0.5">
                    Completed at: {new Date(statusData.completed_at).toLocaleString()}
                  </div>
                )}
              </div>
            </div>
            <Link
              to={`/admin/clients/${activeId}/models`}
              className="px-3 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold text-xs shadow-sm transition-all"
            >
              Inspect Model Registry
            </Link>
          </div>
        )}

        {isFailed && (
          <div className="mt-6 p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs flex items-start gap-2.5">
            <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <span className="font-bold">Training Failed:</span> {statusData?.error_message || 'An error occurred during execution.'}
            </div>
          </div>
        )}
      </div>

      {/* Hyperparameters & Launch Form */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-sm">
        <h2 className="text-sm font-bold text-white mb-4 flex items-center gap-2">
          <Play className="w-4 h-4 text-emerald-400" />
          Training Configuration
        </h2>

        <form onSubmit={handleStartTraining} className="space-y-4 text-xs">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-slate-300 font-semibold mb-1">Epochs</label>
              <input
                type="number"
                min="1"
                max="50"
                value={epochs}
                onChange={(e) => setEpochs(e.target.value)}
                disabled={isRunning}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono focus:border-indigo-500"
              />
              <span className="text-[11px] text-slate-500">Default: 15</span>
            </div>

            <div>
              <label className="block text-slate-300 font-semibold mb-1">Batch Size</label>
              <input
                type="number"
                min="16"
                max="1024"
                step="16"
                value={batchSize}
                onChange={(e) => setBatchSize(e.target.value)}
                disabled={isRunning}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono focus:border-indigo-500"
              />
              <span className="text-[11px] text-slate-500">Default: 256</span>
            </div>

            <div>
              <label className="block text-slate-300 font-semibold mb-1">Learning Rate</label>
              <input
                type="number"
                min="0.00001"
                max="0.1"
                step="any"
                value={learningRate}
                onChange={(e) => setLearningRate(e.target.value)}
                disabled={isRunning}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 font-mono focus:border-indigo-500"
              />
              <span className="text-[11px] text-slate-500">Default: 0.001 (AdamW)</span>
            </div>
          </div>

          <div className="pt-4 border-t border-slate-800 flex items-center justify-between">
            <span className="text-slate-400">
              Training runs asynchronously in the background without blocking the UI.
            </span>
            <button
              type="submit"
              disabled={isRunning || triggering}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-bold text-xs shadow-md shadow-indigo-600/20 transition-all"
            >
              {triggering ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Scheduling...
                </>
              ) : isRunning ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Training Running...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Start Training Run
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
