import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useClient } from '../../context/ClientContext';
import { api } from '../../services/api';
import {
  FileSpreadsheet,
  UploadCloud,
  CheckCircle2,
  AlertTriangle,
  AlertCircle,
  Table,
  ArrowRight,
  Loader2,
  RefreshCw,
} from 'lucide-react';

export default function DatasetUploadPage() {
  const { clientId } = useParams();
  const { selectedClientId, selectClient } = useClient();

  const activeId = clientId || selectedClientId;

  useEffect(() => {
    if (clientId && clientId !== selectedClientId) {
      selectClient(clientId);
    }
  }, [clientId, selectedClientId, selectClient]);

  const [datasetInfo, setDatasetInfo] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  const fetchDatasetDetails = async () => {
    setLoading(true);
    setError(null);
    try {
      const [info, previewData] = await Promise.allSettled([
        api.getDataset(activeId),
        api.getDatasetPreview(activeId, 10, 0),
      ]);

      if (info.status === 'fulfilled') {
        setDatasetInfo(info.value);
      } else {
        setDatasetInfo(null);
      }

      if (previewData.status === 'fulfilled') {
        setPreview(previewData.value);
      } else {
        setPreview(null);
      }
    } catch (err) {
      console.error('Failed to load dataset details:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatasetDetails();
  }, [activeId]);

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setError(null);
    setSuccessMsg(null);

    try {
      const resp = await api.uploadDataset(activeId, file);
      setSuccessMsg(`Dataset '${file.name}' successfully uploaded and verified.`);
      await fetchDatasetDetails();
    } catch (err) {
      setError(err.message || 'File upload failed.');
    } finally {
      setUploading(false);
    }
  };

  const hasTimestamp = preview?.has_timestamp !== false && datasetInfo?.has_timestamp !== false;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
            <FileSpreadsheet className="w-6 h-6 text-emerald-400" />
            Dataset Ingestion & Inspection
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Client: <span className="font-mono text-emerald-400 font-semibold">{activeId}</span>
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={fetchDatasetDetails}
            disabled={loading}
            className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
            title="Refresh Inspection"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <Link
            to={`/admin/clients/${activeId}/schema`}
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold text-xs shadow-sm transition-all"
          >
            Proceed to Schema Mapping
            <ArrowRight className="w-3.5 h-3.5" />
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

      {/* Upload Zone */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 sm:p-8 shadow-sm">
        <div className="border-2 border-dashed border-slate-700 hover:border-emerald-500/60 rounded-xl p-8 text-center transition-colors">
          <input
            type="file"
            id="dataset-upload-input"
            accept=".csv,.parquet"
            onChange={handleFileUpload}
            disabled={uploading}
            className="hidden"
          />
          <label
            htmlFor="dataset-upload-input"
            className="cursor-pointer flex flex-col items-center justify-center space-y-3"
          >
            <div className="w-12 h-12 rounded-full bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
              {uploading ? (
                <Loader2 className="w-6 h-6 animate-spin" />
              ) : (
                <UploadCloud className="w-6 h-6" />
              )}
            </div>
            <div>
              <div className="text-sm font-bold text-white">
                {uploading ? 'Processing & validating dataset...' : 'Click to upload or drag & drop dataset'}
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Supports CSV or Parquet files. Must contain customer interactions with identifier columns.
              </p>
            </div>
            <span className="px-3 py-1 rounded-md bg-slate-800 text-[11px] font-mono text-slate-300 border border-slate-700">
              Target: clients/{activeId}/data/raw/
            </span>
          </label>
        </div>
      </div>

      {/* Dataset Metadata Inspection Card */}
      {datasetInfo && (
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
            <h2 className="text-sm font-bold text-white flex items-center gap-2">
              <Table className="w-4 h-4 text-emerald-400" />
              Ingested Dataset Telemetry
            </h2>
            {/* Timestamp Policy Badge */}
            {hasTimestamp ? (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <CheckCircle2 className="w-3.5 h-3.5" />
                Temporal Capability Enabled (Zero-Leakage Active)
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20">
                <AlertTriangle className="w-3.5 h-3.5" />
                Timestamp Column Absent: Temporal Capability Disabled
              </span>
            )}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <div className="text-slate-400">Total Rows</div>
              <div className="text-base font-bold text-white mt-1">
                {datasetInfo.row_count?.toLocaleString() || '—'}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <div className="text-slate-400">Total Columns</div>
              <div className="text-base font-bold text-white mt-1">
                {datasetInfo.column_count || '—'}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <div className="text-slate-400">Storage Size</div>
              <div className="text-base font-bold text-white mt-1">
                {datasetInfo.file_size_mb ? `${datasetInfo.file_size_mb.toFixed(2)} MB` : '—'}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <div className="text-slate-400">Chronological Split</div>
              <div className="text-base font-bold text-white mt-1">
                {hasTimestamp ? 'Strict Point-in-Time' : 'Random Stratified'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Live Preview Table */}
      {preview?.rows && preview.rows.length > 0 && (
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
          <div className="p-4 bg-slate-950/80 border-b border-slate-800 flex items-center justify-between">
            <div className="text-xs font-bold text-white flex items-center gap-2">
              <span>Interactive Data Sample (First 10 Rows)</span>
              <span className="text-[11px] text-slate-400 font-mono">
                Total Columns: {preview.columns?.length}
              </span>
            </div>
          </div>
          <div className="overflow-x-auto max-h-96">
            <table className="w-full text-left text-xs whitespace-nowrap">
              <thead className="bg-slate-950/90 text-slate-400 font-mono text-[11px] uppercase border-b border-slate-800 sticky top-0">
                <tr>
                  {preview.columns?.map((col) => (
                    <th key={col} className="py-2.5 px-3 font-semibold">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                {preview.rows.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/40">
                    {preview.columns?.map((col) => (
                      <td key={col} className="py-2 px-3 text-slate-300">
                        {row[col] !== null && row[col] !== undefined ? String(row[col]) : (
                          <span className="text-slate-600 italic">null</span>
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
