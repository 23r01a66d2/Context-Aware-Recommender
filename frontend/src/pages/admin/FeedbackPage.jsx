import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useClient } from '../../context/ClientContext';
import { api } from '../../services/api';
import {
  MessageSquare,
  CheckCircle2,
  XCircle,
  MousePointer,
  ShoppingBag,
  RefreshCw,
  Search,
  Filter,
} from 'lucide-react';

export default function FeedbackPage() {
  const { clientId } = useParams();
  const { selectedClientId, selectClient } = useClient();
  const activeId = clientId || selectedClientId;

  useEffect(() => {
    if (clientId && clientId !== selectedClientId) {
      selectClient(clientId);
    }
  }, [clientId, selectedClientId, selectClient]);

  const [feedbackList, setFeedbackList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filterAction, setFilterAction] = useState('ALL');
  const [searchUser, setSearchUser] = useState('');

  const fetchFeedback = async () => {
    setLoading(true);
    try {
      const data = await api.getFeedbackLog(activeId);
      setFeedbackList(data);
    } catch (err) {
      console.error('Failed to load feedback log:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFeedback();
  }, [activeId]);

  const filtered = feedbackList.filter((fb) => {
    const matchAction = filterAction === 'ALL' || fb.action === filterAction;
    const matchUser = !searchUser || String(fb.user_id).toLowerCase().includes(searchUser.toLowerCase());
    return matchAction && matchUser;
  });

  const getActionBadge = (action) => {
    switch (action) {
      case 'CLICK':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-sky-500/10 text-sky-400 border border-sky-500/20">
            <MousePointer className="w-3 h-3" /> CLICK
          </span>
        );
      case 'ACCEPT':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <CheckCircle2 className="w-3 h-3" /> ACCEPT
          </span>
        );
      case 'REJECT':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20">
            <XCircle className="w-3 h-3" /> REJECT
          </span>
        );
      case 'PURCHASE':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <ShoppingBag className="w-3 h-3" /> PURCHASE
          </span>
        );
      default:
        return (
          <span className="px-2 py-0.5 rounded text-[11px] bg-slate-800 text-slate-300">
            {action}
          </span>
        );
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
            <MessageSquareCheck className="w-6 h-6 text-emerald-400" />
            Feedback Event Audit Log
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Client: <span className="font-mono text-emerald-400 font-semibold">{activeId}</span>
          </p>
        </div>

        <button
          onClick={fetchFeedback}
          disabled={loading}
          className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 transition-colors self-start sm:self-auto"
          title="Refresh Log"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Filter / Search Bar */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <Filter className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-slate-400">Action:</span>
          <select
            value={filterAction}
            onChange={(e) => setFilterAction(e.target.value)}
            className="bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-slate-200 focus:outline-none focus:border-emerald-500"
          >
            <option value="ALL">All Actions ({feedbackList.length})</option>
            <option value="CLICK">CLICK</option>
            <option value="ACCEPT">ACCEPT</option>
            <option value="REJECT">REJECT</option>
            <option value="PURCHASE">PURCHASE</option>
          </select>
        </div>

        <div className="relative w-full sm:w-64">
          <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search by User ID..."
            value={searchUser}
            onChange={(e) => setSearchUser(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-8 pr-3 py-1.5 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
          />
        </div>
      </div>

      {/* Table */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/80 border-b border-slate-800 text-slate-400 font-mono text-[11px] uppercase">
              <tr>
                <th className="py-3 px-4">Event ID</th>
                <th className="py-3 px-4">Timestamp</th>
                <th className="py-3 px-4">User ID</th>
                <th className="py-3 px-4">Item ID</th>
                <th className="py-3 px-4">Action</th>
                <th className="py-3 px-4">Linked Recommendation ID</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
              {loading ? (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-slate-500 font-sans">
                    Loading feedback log...
                  </td>
                </tr>
              ) : filtered.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-slate-500 font-sans">
                    No feedback records found. Go to Live Recommendations and interact with items to log events.
                  </td>
                </tr>
              ) : (
                filtered.map((fb) => (
                  <tr key={fb.id || `${fb.user_id}-${fb.item_id}-${fb.timestamp}`} className="hover:bg-slate-800/40">
                    <td className="py-3 px-4 text-slate-400">#{fb.id}</td>
                    <td className="py-3 px-4 text-slate-400">
                      {fb.timestamp ? new Date(fb.timestamp).toLocaleString() : '—'}
                    </td>
                    <td className="py-3 px-4 text-slate-200 font-semibold">{fb.user_id}</td>
                    <td className="py-3 px-4 text-emerald-400">{fb.item_id}</td>
                    <td className="py-3 px-4">{getActionBadge(fb.action)}</td>
                    <td className="py-3 px-4 text-slate-500">
                      {fb.recommendation_id ? (
                        <span className="text-slate-300 font-mono bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800">
                          {fb.recommendation_id}
                        </span>
                      ) : (
                        <span className="italic text-slate-600">unlinked</span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
