import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useClient } from '../../context/ClientContext';
import { api } from '../../services/api';
import {
  Building2,
  Plus,
  ArrowRight,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Trash2,
  AlertTriangle,
  ShieldCheck,
  X
} from 'lucide-react';

export default function ClientsPage() {
  const { clients, selectedClientId, selectClient, handleClientDeleted, refreshClients, loadingClients } = useClient();
  const [refreshing, setRefreshing] = useState(false);

  // Deletion Modal State
  const [clientToDelete, setClientToDelete] = useState(null);
  const [typedConfirmId, setTypedConfirmId] = useState('');
  const [deletePhysicalFiles, setDeletePhysicalFiles] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState(null);
  const [notification, setNotification] = useState(null);

  const handleManualRefresh = async () => {
    setRefreshing(true);
    await refreshClients();
    setRefreshing(false);
  };

  const openDeleteModal = (client) => {
    setClientToDelete(client);
    setTypedConfirmId('');
    setDeletePhysicalFiles(false);
    setDeleteError(null);
  };

  const closeDeleteModal = () => {
    if (deleting) return;
    setClientToDelete(null);
    setTypedConfirmId('');
    setDeletePhysicalFiles(false);
    setDeleteError(null);
  };

  const handleDeleteClient = async () => {
    if (!clientToDelete || typedConfirmId !== clientToDelete.client_id) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await api.deleteClient(clientToDelete.client_id, deletePhysicalFiles);
      if (handleClientDeleted) {
        handleClientDeleted(clientToDelete.client_id);
      }
      setNotification({
        type: 'success',
        message: `Client "${clientToDelete.client_id}" was deleted successfully.`
      });
      setClientToDelete(null);
      setTypedConfirmId('');
      setDeletePhysicalFiles(false);
      await refreshClients();
    } catch (err) {
      setDeleteError(err.message || 'Failed to delete client');
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Toast Notification */}
      {notification && (
        <div className="flex items-center justify-between p-3.5 rounded-xl bg-teal-500/10 border border-teal-500/30 text-teal-700 dark:text-teal-300 text-xs animate-fadeIn">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-teal-500 flex-shrink-0" />
            <span>{notification.message}</span>
          </div>
          <button
            onClick={() => setNotification(null)}
            className="text-teal-600 dark:text-teal-300 hover:text-teal-800 dark:hover:text-teal-100 transition-colors ml-4 cursor-pointer"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold dark:text-coolwhite text-slate-900 tracking-tight flex items-center gap-2">
            <Building2 className="w-5 h-5 text-teal-500" />
            Client Organizations
          </h1>
          <p className="text-xs dark:text-muted text-slate-500 mt-0.5">
            Manage multi-tenant isolated workspaces, schema definitions, and model versions.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleManualRefresh}
            disabled={refreshing}
            className="p-2 rounded-xl dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 dark:text-muted text-slate-600 hover:dark:text-coolwhite hover:text-slate-900 transition-colors cursor-pointer shadow-sm"
            title="Refresh Client List"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
          </button>
          <Link
            to="/admin/clients/new"
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-teal-500 hover:bg-teal-400 text-navy-950 font-bold text-xs shadow-sm transition-all"
          >
            <Plus className="w-4 h-4" />
            Add New Client
          </Link>
        </div>
      </div>

      {/* Clients Table */}
      <div className="dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 rounded-2xl overflow-hidden shadow-sm transition-colors duration-200">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="dark:bg-navy-950/80 bg-slate-50 border-b dark:border-navy-800 border-slate-200 dark:text-muted text-slate-500 uppercase font-semibold">
              <tr>
                <th className="py-3.5 px-4">Client ID</th>
                <th className="py-3.5 px-4">Organization Name</th>
                <th className="py-3.5 px-4">Active Model</th>
                <th className="py-3.5 px-4">Status</th>
                <th className="py-3.5 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y dark:divide-navy-800/60 divide-slate-100">
              {loadingClients ? (
                <tr>
                  <td colSpan={5} className="py-8 text-center dark:text-muted text-slate-500">
                    Loading clients...
                  </td>
                </tr>
              ) : clients.length === 0 ? (
                <tr>
                  <td colSpan={5} className="py-8 text-center dark:text-muted text-slate-500">
                    No clients found. Click "Add New Client" to create one.
                  </td>
                </tr>
              ) : (
                clients.map((client) => {
                  const isSelected = client.client_id === selectedClientId;
                  const isSystem = Boolean(client.is_system);

                  return (
                    <tr
                      key={client.client_id}
                      className={`dark:hover:bg-navy-850 hover:bg-slate-50 transition-colors ${
                        isSelected ? 'dark:bg-teal-500/[0.04] bg-teal-50/50' : ''
                      }`}
                    >
                      <td className="py-3.5 px-4 font-mono font-medium dark:text-coolwhite text-slate-900">
                        {client.client_id}
                        {isSelected && (
                          <span className="ml-2 text-[10px] px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-400 font-sans border border-teal-300 dark:border-teal-500/20 font-bold">
                            Active Scope
                          </span>
                        )}
                      </td>
                      <td className="py-3.5 px-4 font-medium dark:text-coolwhite text-slate-900">{client.name || '—'}</td>
                      <td className="py-3.5 px-4">
                        {client.active_model_version ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-mono font-semibold bg-teal-500/10 text-teal-700 dark:text-teal-400 border border-teal-300 dark:border-teal-500/20">
                            <CheckCircle2 className="w-3 h-3 text-teal-500" />
                            {client.active_model_version}
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] dark:bg-navy-950 bg-slate-100 dark:text-muted text-slate-500 border dark:border-navy-800 border-slate-200">
                            <AlertCircle className="w-3 h-3" />
                            No Active Model
                          </span>
                        )}
                      </td>
                      <td className="py-3.5 px-4">
                        {isSystem ? (
                          <span
                            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-300 dark:border-amber-500/20 font-medium"
                            title="Reference benchmark client protected against deletion"
                          >
                            <ShieldCheck className="w-3 h-3 text-amber-500" />
                            Protected Reference
                          </span>
                        ) : (
                          <span className="dark:text-muted text-slate-600">Ready</span>
                        )}
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <div className="flex items-center justify-end gap-2">
                          {!isSelected && (
                            <button
                              onClick={() => selectClient(client.client_id)}
                              className="px-2.5 py-1 rounded-lg dark:bg-navy-800 bg-slate-100 hover:dark:bg-navy-700 hover:bg-slate-200 dark:text-coolwhite text-slate-800 font-medium text-[11px] transition-colors cursor-pointer"
                            >
                              Switch Scope
                            </button>
                          )}
                          <Link
                            to={`/admin/clients/${client.client_id}/dataset`}
                            onClick={() => selectClient(client.client_id)}
                            className="p-1.5 rounded-lg hover:dark:bg-navy-800 hover:bg-slate-100 dark:text-muted text-slate-500 hover:text-teal-500 transition-colors"
                            title="Manage Dataset & Models"
                          >
                            <ArrowRight className="w-4 h-4" />
                          </Link>

                          {/* Delete Action */}
                          {isSystem ? (
                            <button
                              disabled
                              className="p-1.5 rounded-lg text-slate-400 cursor-not-allowed opacity-40"
                              title="Reference benchmark client cannot be deleted"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          ) : (
                            <button
                              onClick={() => openDeleteModal(client)}
                              className="p-1.5 rounded-lg hover:bg-rose-500/15 text-rose-500 dark:text-rose-400 transition-colors cursor-pointer"
                              title={`Delete ${client.client_id}`}
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Confirmation Modal */}
      {clientToDelete && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="dark:bg-navy-900 bg-white border border-rose-500/40 rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-2xl relative animate-scaleUp">
            <button
              onClick={closeDeleteModal}
              disabled={deleting}
              className="absolute top-4 right-4 dark:text-muted text-slate-400 hover:dark:text-coolwhite hover:text-slate-800 transition-colors cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-start gap-3">
              <div className="p-2.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-500 flex-shrink-0">
                <AlertTriangle className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-base font-bold dark:text-coolwhite text-slate-900">Delete Client Organization</h3>
                <p className="text-xs dark:text-muted text-slate-500 mt-0.5">
                  This action permanently removes the client registration and all associated platform records.
                </p>
              </div>
            </div>

            {/* Client details card */}
            <div className="dark:bg-navy-950 bg-slate-50 rounded-xl p-3 border dark:border-navy-800 border-slate-200 text-xs space-y-1.5">
              <div className="flex justify-between">
                <span className="dark:text-muted text-slate-500">Client ID:</span>
                <span className="font-mono font-bold text-rose-500 dark:text-rose-400">{clientToDelete.client_id}</span>
              </div>
              <div className="flex justify-between">
                <span className="dark:text-muted text-slate-500">Organization Name:</span>
                <span className="dark:text-coolwhite text-slate-800 font-medium">{clientToDelete.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="dark:text-muted text-slate-500">Active Model:</span>
                <span className="dark:text-coolwhite text-slate-800 font-mono">{clientToDelete.active_model_version || 'None'}</span>
              </div>
            </div>

            {/* Impact List */}
            <div className="text-xs dark:text-muted text-slate-600 space-y-1">
              <p className="font-semibold dark:text-coolwhite text-slate-800">The following records will be permanently removed:</p>
              <ul className="list-disc list-inside space-y-0.5 pl-1">
                <li>Platform client registration</li>
                <li>Schema mappings & inferred feature configs</li>
                <li>Dataset registrations & preview metadata</li>
                <li>Model version registry entries & training history</li>
                <li>Logged recommendation & feedback telemetry</li>
              </ul>
            </div>

            {/* Physical files checkbox */}
            <div className="dark:bg-navy-950 bg-slate-50 p-3 rounded-xl border dark:border-navy-800 border-slate-200">
              <label className="flex items-start gap-2.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={deletePhysicalFiles}
                  onChange={(e) => setDeletePhysicalFiles(e.target.checked)}
                  disabled={deleting}
                  className="mt-0.5 rounded border-slate-300 dark:border-navy-700 text-rose-600 focus:ring-rose-500 dark:bg-navy-900 w-4 h-4 cursor-pointer"
                />
                <div className="text-xs">
                  <span className="font-medium dark:text-coolwhite text-slate-800">
                    Also permanently delete local files and model checkpoints on disk
                  </span>
                  <p className="text-[11px] dark:text-muted text-slate-500 mt-0.5">
                    Target folders: <code className="dark:text-coolwhite text-slate-800 font-mono">clients/{clientToDelete.client_id}</code> and <code className="dark:text-coolwhite text-slate-800 font-mono">models/{clientToDelete.client_id}</code>. If unchecked, files are preserved for re-registration.
                  </p>
                </div>
              </label>
            </div>

            {/* Typed confirmation input */}
            <div className="space-y-1.5">
              <label className="block text-xs font-medium dark:text-coolwhite text-slate-800">
                To confirm deletion, please type <span className="font-mono text-rose-500 dark:text-rose-400 font-bold">{clientToDelete.client_id}</span> below:
              </label>
              <input
                type="text"
                value={typedConfirmId}
                onChange={(e) => setTypedConfirmId(e.target.value)}
                disabled={deleting}
                placeholder={clientToDelete.client_id}
                className="w-full px-3 py-2 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-300 dark:text-coolwhite text-slate-900 font-mono text-xs focus:outline-none focus:border-rose-500 transition-colors"
              />
            </div>

            {deleteError && (
              <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-600 dark:text-rose-400 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{deleteError}</span>
              </div>
            )}

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={closeDeleteModal}
                disabled={deleting}
                className="px-4 py-2 rounded-xl dark:bg-navy-800 bg-slate-100 hover:dark:bg-navy-700 hover:bg-slate-200 dark:text-coolwhite text-slate-800 font-medium text-xs transition-colors cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleDeleteClient}
                disabled={typedConfirmId !== clientToDelete.client_id || deleting}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-700 disabled:opacity-40 disabled:cursor-not-allowed text-white font-bold text-xs transition-all shadow-sm cursor-pointer"
              >
                {deleting ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 className="w-3.5 h-3.5" />
                    Delete Client
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
