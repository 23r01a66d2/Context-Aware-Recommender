import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useClient } from '../../context/ClientContext';
import { api } from '../../services/api';
import { Building2, ArrowLeft, Check, AlertCircle, Loader2 } from 'lucide-react';

export default function NewClientPage() {
  const navigate = useNavigate();
  const { selectClient, refreshClients } = useClient();

  const [clientId, setClientId] = useState('');
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    const cleanId = clientId.trim().toLowerCase().replace(/[^a-z0-9_]/g, '_');
    if (!cleanId) {
      setError('Please provide a valid Client ID (alphanumeric and underscores only).');
      return;
    }

    setLoading(true);
    try {
      await api.createClient({
        client_id: cleanId,
        name: name.trim() || cleanId,
      });

      await refreshClients();
      selectClient(cleanId);
      navigate(`/admin/clients/${cleanId}/dataset`);
    } catch (err) {
      setError(err.message || 'Failed to create client organization.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <Link
        to="/admin/clients"
        className="inline-flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200 transition-colors"
      >
        <ArrowLeft className="w-3.5 h-3.5" /> Back to Clients
      </Link>

      <div className="bg-white dark:bg-navy-900/90 border border-slate-200 dark:border-navy-800 rounded-xl p-6 sm:p-8 shadow-sm">
        <div className="flex items-center gap-3 border-b border-slate-100 dark:border-navy-800 pb-5 mb-6">
          <div className="w-10 h-10 rounded-lg bg-teal-500/10 border border-teal-500/20 flex items-center justify-center text-teal-600 dark:text-teal-400">
            <Building2 className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-900 dark:text-white">Create New Client Organization</h1>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Provisions a dedicated directory structure, database records, and schema space.
            </p>
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 rounded-lg bg-rose-50 dark:bg-rose-950/70 border border-rose-200 dark:border-rose-800/80 text-rose-700 dark:text-rose-300 text-xs flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-500 dark:text-rose-400" />
            <div>
              <span className="font-semibold">Creation Error:</span> {error}
            </div>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-5 text-xs">
          <div>
            <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">
              Client ID (Tenant Identifier) <span className="text-rose-500 dark:text-rose-400">*</span>
            </label>
            <input
              type="text"
              value={clientId}
              onChange={(e) => setClientId(e.target.value)}
              placeholder="e.g. fashion_retail_london"
              required
              className="w-full bg-slate-50 dark:bg-navy-950 border border-slate-200 dark:border-navy-800 rounded-lg px-3.5 py-2.5 text-slate-900 dark:text-coolwhite font-mono focus:outline-none focus:border-teal-500 dark:focus:border-teal-400 transition-colors"
            />
            <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
              Must be unique across the platform. Alphanumeric characters and underscores only.
            </p>
          </div>

          <div>
            <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">
              Display Name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. London Flagship Fashion"
              className="w-full bg-slate-50 dark:bg-navy-950 border border-slate-200 dark:border-navy-800 rounded-lg px-3.5 py-2.5 text-slate-900 dark:text-coolwhite focus:outline-none focus:border-teal-500 dark:focus:border-teal-400 transition-colors"
            />
          </div>

          <div className="pt-4 border-t border-slate-100 dark:border-navy-800 flex items-center justify-end gap-3">
            <Link
              to="/admin/clients"
              className="px-4 py-2 rounded-lg bg-slate-100 dark:bg-navy-800 hover:bg-slate-200 dark:hover:bg-navy-700 text-slate-700 dark:text-slate-300 font-medium transition-colors"
            >
              Cancel
            </Link>
            <button
              type="submit"
              disabled={loading}
              className="inline-flex items-center gap-2 px-5 py-2 rounded-lg bg-teal-500 hover:bg-teal-600 disabled:opacity-50 text-slate-950 font-bold transition-all shadow-sm"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Provisioning...
                </>
              ) : (
                <>
                  <Check className="w-4 h-4" />
                  Create Client
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
