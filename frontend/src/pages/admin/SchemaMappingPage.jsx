import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useClient } from '../../context/ClientContext';
import { api } from '../../services/api';
import {
  Network,
  CheckCircle2,
  AlertTriangle,
  AlertCircle,
  Save,
  ArrowRight,
  Loader2,
  RefreshCw,
  Sliders,
} from 'lucide-react';
import { CANONICAL_SCHEMA_ROLES, CANONICAL_DATA_TYPES } from '../../constants/schemaRoles';

export default function SchemaMappingPage() {
  const { clientId } = useParams();
  const { selectedClientId, selectClient } = useClient();
  const activeId = clientId || selectedClientId;

  useEffect(() => {
    if (clientId && clientId !== selectedClientId) {
      selectClient(clientId);
    }
  }, [clientId, selectedClientId, selectClient]);

  const [schemaData, setSchemaData] = useState(null);
  const [columnMappings, setColumnMappings] = useState({});
  const [columnTypes, setColumnTypes] = useState({});
  const [coldStartThreshold, setColdStartThreshold] = useState(3);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  const fetchSchema = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getSchema(activeId);
      setSchemaData(data);
      setColumnMappings(data.column_mappings || {});
      setColumnTypes(data.column_types || {});
      setColdStartThreshold(data.cold_start_threshold ?? 3);
    } catch (err) {
      setError(err.message || 'Failed to load client schema mapping.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSchema();
  }, [activeId]);

  const handleRoleChange = (col, newRole) => {
    setColumnMappings((prev) => ({ ...prev, [col]: newRole }));
  };

  const handleTypeChange = (col, newType) => {
    setColumnTypes((prev) => ({ ...prev, [col]: newType }));
  };

  // Validation Logic
  const mappedUser = Object.entries(columnMappings).find(([_, r]) => r === 'USER_ID')?.[0];
  const mappedItem = Object.entries(columnMappings).find(([_, r]) => r === 'ITEM_ID')?.[0];
  const mappedTimestamp = Object.entries(columnMappings).find(([_, r]) => r === 'TIMESTAMP')?.[0];
  const contentCount = Object.values(columnMappings).filter((r) => r === 'CONTENT_FEATURE').length;
  const contextCount = Object.values(columnMappings).filter((r) => r === 'CONTEXT_FEATURE').length;

  const isValidForTraining = Boolean(mappedUser && mappedItem && (contentCount > 0 || contextCount > 0));

  const handleSave = async (e) => {
    e.preventDefault();
    if (!mappedUser || !mappedItem) {
      setError('USER_ID and ITEM_ID are mandatory for recommendation modeling.');
      return;
    }

    setSaving(true);
    setError(null);
    setSuccessMsg(null);

    try {
      await api.saveSchema(activeId, {
        column_mappings: columnMappings,
        column_types: columnTypes,
        cold_start_threshold: Number(coldStartThreshold),
        temporal_capability: Boolean(mappedTimestamp),
      });
      setSuccessMsg('Schema configuration successfully validated and saved.');
      await fetchSchema();
    } catch (err) {
      setError(err.message || 'Failed to save schema mapping.');
    } finally {
      setSaving(false);
    }
  };

  const allColumns = Object.keys(columnMappings);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
            <Network className="w-6 h-6 text-teal-400" />
            Schema Definition & Feature Roles
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Client: <span className="font-mono text-emerald-400 font-semibold">{activeId}</span>
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={fetchSchema}
            disabled={loading}
            className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
            title="Reload Schema"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <Link
            to={`/admin/clients/${activeId}/training`}
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold text-xs shadow-sm transition-all"
          >
            Go to Model Training
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </div>

      {/* Status Badges & Validation Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
        <div
          className={`p-3 rounded-xl border text-xs ${
            mappedUser
              ? 'bg-emerald-950/40 border-emerald-800/60 text-emerald-300'
              : 'bg-rose-950/40 border-rose-800/60 text-rose-300'
          }`}
        >
          <div className="font-semibold flex items-center gap-1.5">
            {mappedUser ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertCircle className="w-3.5 h-3.5" />}
            USER_ID
          </div>
          <div className="text-[11px] mt-1 text-slate-400 truncate">
            {mappedUser ? `Mapped to '${mappedUser}'` : 'Missing (Mandatory)'}
          </div>
        </div>

        <div
          className={`p-3 rounded-xl border text-xs ${
            mappedItem
              ? 'bg-emerald-950/40 border-emerald-800/60 text-emerald-300'
              : 'bg-rose-950/40 border-rose-800/60 text-rose-300'
          }`}
        >
          <div className="font-semibold flex items-center gap-1.5">
            {mappedItem ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertCircle className="w-3.5 h-3.5" />}
            ITEM_ID
          </div>
          <div className="text-[11px] mt-1 text-slate-400 truncate">
            {mappedItem ? `Mapped to '${mappedItem}'` : 'Missing (Mandatory)'}
          </div>
        </div>

        <div
          className={`p-3 rounded-xl border text-xs ${
            mappedTimestamp
              ? 'bg-emerald-950/40 border-emerald-800/60 text-emerald-300'
              : 'bg-amber-950/40 border-amber-800/60 text-amber-300'
          }`}
        >
          <div className="font-semibold flex items-center gap-1.5">
            {mappedTimestamp ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertTriangle className="w-3.5 h-3.5" />}
            TIMESTAMP
          </div>
          <div className="text-[11px] mt-1 text-slate-400 truncate">
            {mappedTimestamp ? `Mapped to '${mappedTimestamp}'` : 'Absent (Temporal Disabled)'}
          </div>
        </div>

        <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 text-xs">
          <div className="font-semibold text-slate-200 flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-purple-400" />
            Feature Modalities
          </div>
          <div className="text-[11px] mt-1 text-slate-400">
            Content: <span className="text-teal-400 font-semibold">{contentCount}</span> | Context:{' '}
            <span className="text-purple-400 font-semibold">{contextCount}</span>
          </div>
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

      {/* Schema Mapping Table */}
      <form onSubmit={handleSave} className="space-y-6">
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
          <div className="p-4 bg-slate-950/80 border-b border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="text-xs font-bold text-white">
              Raw Column to Semantic Role & DataType Mapping
            </div>

            <div className="flex items-center gap-3">
              <label className="text-xs text-slate-400 font-medium">
                Cold-Start Threshold (K):
              </label>
              <input
                type="number"
                min="1"
                max="20"
                value={coldStartThreshold}
                onChange={(e) => setColdStartThreshold(e.target.value)}
                className="w-16 bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs text-slate-200 text-center font-mono focus:border-emerald-500"
              />
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/90 text-slate-400 font-mono text-[11px] uppercase border-b border-slate-800">
                <tr>
                  <th className="py-3 px-4">Dataset Column</th>
                  <th className="py-3 px-4">Semantic Role</th>
                  <th className="py-3 px-4">Data Type</th>
                  <th className="py-3 px-4">Transformation Pipeline Target</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {allColumns.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="py-8 text-center text-slate-500 font-sans">
                      No columns detected. Please upload an interaction dataset first.
                    </td>
                  </tr>
                ) : (
                  allColumns.map((col) => {
                    const currentRole = columnMappings[col] || 'IGNORE';
                    const currentType = columnTypes[col] || 'categorical';

                    return (
                      <tr key={col} className="hover:bg-slate-800/40 transition-colors">
                        <td className="py-3 px-4 font-semibold text-slate-200">{col}</td>
                        <td className="py-3 px-4">
                          <select
                            value={currentRole}
                            onChange={(e) => handleRoleChange(col, e.target.value)}
                            className="bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-emerald-500 cursor-pointer"
                          >
                            {CANONICAL_SCHEMA_ROLES.map((r) => (
                              <option key={r.value} value={r.value}>
                                {r.label}
                              </option>
                            ))}
                          </select>
                        </td>
                        <td className="py-3 px-4">
                          <select
                            value={currentType}
                            onChange={(e) => handleTypeChange(col, e.target.value)}
                            className="bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-emerald-500 cursor-pointer"
                          >
                            {CANONICAL_DATA_TYPES.map((t) => (
                              <option key={t} value={t}>
                                {t}
                              </option>
                            ))}
                          </select>
                        </td>
                        <td className="py-3 px-4 font-sans text-slate-400 text-[11px]">
                          {currentRole === 'USER_ID' && 'Behavioral History Grouping'}
                          {currentRole === 'ITEM_ID' && 'Candidate Pool & Embedding ID'}
                          {currentRole === 'TIMESTAMP' && 'Chronological Point-in-Time Split'}
                          {currentRole === 'TARGET' && 'BCE Supervised Relevance Label'}
                          {currentRole === 'BEHAVIOR_FEATURE' && 'Behavioral History Encoder Projection'}
                          {currentRole === 'CONTENT_FEATURE' && 'Content Tower Projection'}
                          {currentRole === 'CONTEXT_FEATURE' && 'Session Context Projection'}
                          {currentRole === 'IGNORE' && 'Filtered Out (Unused / Leakage Prevention)'}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          <div className="p-4 bg-slate-950/80 border-t border-slate-800 flex items-center justify-between">
            <span className="text-xs text-slate-400 font-sans">
              Ensure <code className="text-emerald-400">USER_ID</code> and{' '}
              <code className="text-emerald-400">ITEM_ID</code> are selected before saving.
            </span>
            <button
              type="submit"
              disabled={saving || !isValidForTraining}
              className="inline-flex items-center gap-2 px-5 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-600 disabled:opacity-40 text-slate-950 font-bold text-xs shadow-sm transition-all"
            >
              {saving ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Saving Schema...
                </>
              ) : (
                <>
                  <Save className="w-4 h-4" />
                  Save & Confirm Schema
                </>
              )}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
