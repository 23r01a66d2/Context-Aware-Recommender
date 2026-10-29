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
  ShoppingCart,
  Eye,
  EyeOff,
  Radio,
  Clock,
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
      case 'IMPRESSION':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700">
            <Radio className="w-3 h-3 text-slate-500 dark:text-slate-400" /> IMPRESSION
          </span>
        );
      case 'VIEW':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-teal-50 dark:bg-teal-500/10 text-teal-700 dark:text-teal-400 border border-teal-200 dark:border-teal-500/20">
            <Eye className="w-3 h-3" /> VIEW
          </span>
        );
      case 'LONG_VIEW':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-indigo-50 dark:bg-indigo-500/10 text-indigo-700 dark:text-indigo-400 border border-indigo-200 dark:border-indigo-500/20">
            <Clock className="w-3 h-3" /> LONG_VIEW
          </span>
        );
      case 'CLICK':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-sky-50 dark:bg-sky-500/10 text-sky-700 dark:text-sky-400 border border-sky-200 dark:border-sky-500/20">
            <MousePointer className="w-3 h-3" /> CLICK
          </span>
        );
      case 'IGNORE':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-200 dark:border-amber-500/20">
            <EyeOff className="w-3 h-3" /> IGNORE
          </span>
        );
      case 'ADD_TO_CART':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-500/20">
            <ShoppingCart className="w-3 h-3" /> ADD_TO_CART
          </span>
        );
      case 'ACCEPT':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-teal-50 dark:bg-emerald-500/10 text-teal-700 dark:text-emerald-400 border border-teal-200 dark:border-emerald-500/20">
            <CheckCircle2 className="w-3 h-3" /> ACCEPT
          </span>
        );
      case 'REJECT':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-400 border border-rose-200 dark:border-rose-500/20">
            <XCircle className="w-3 h-3" /> REJECT
          </span>
        );
      case 'PURCHASE':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-purple-50 dark:bg-purple-500/10 text-purple-700 dark:text-purple-400 border border-purple-200 dark:border-purple-500/20">
            <ShoppingBag className="w-3 h-3" /> PURCHASE
          </span>
        );
      default:
        return (
          <span className="px-2 py-0.5 rounded text-[11px] bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
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
          <h1 className="text-xl sm:text-2xl font-extrabold text-slate-900 dark:text-white tracking-tight flex items-center gap-2">
            <MessageSquare className="w-6 h-6 text-teal-600 dark:text-teal-400" />
            Feedback Event Audit Log
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Client: <span className="font-mono text-teal-600 dark:text-teal-400 font-semibold">{activeId}</span>
          </p>
        </div>

        <button
          onClick={fetchFeedback}
          disabled={loading}
          className="p-2 rounded-lg bg-white dark:bg-navy-900 border border-slate-200 dark:border-navy-800 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 transition-colors self-start sm:self-auto shadow-sm"
          title="Refresh Log"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Filter / Search Bar */}
      <div className="bg-white dark:bg-navy-900/90 border border-slate-200 dark:border-navy-800 rounded-xl p-4 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs shadow-sm">
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <Filter className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400" />
          <span className="text-slate-600 dark:text-slate-400 font-medium">Action:</span>
          <select
            value={filterAction}
            onChange={(e) => setFilterAction(e.target.value)}
            className="bg-slate-50 dark:bg-navy-950 border border-slate-200 dark:border-navy-800 rounded px-2.5 py-1 text-slate-900 dark:text-slate-200 focus:outline-none focus:border-teal-500 cursor-pointer"
          >
            <option value="ALL">All Actions ({feedbackList.length})</option>
            <option value="IMPRESSION">IMPRESSION</option>
            <option value="VIEW">VIEW</option>
            <option value="LONG_VIEW">LONG_VIEW</option>
            <option value="CLICK">CLICK</option>
            <option value="IGNORE">IGNORE</option>
            <option value="ADD_TO_CART">ADD_TO_CART</option>
            <option value="PURCHASE">PURCHASE</option>
            <option value="ACCEPT">ACCEPT</option>
            <option value="REJECT">REJECT</option>
          </select>
        </div>

        <div className="relative w-full sm:w-64">
          <Search className="w-3.5 h-3.5 text-slate-400 dark:text-slate-500 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search by User ID..."
            value={searchUser}
            onChange={(e) => setSearchUser(e.target.value)}
            className="w-full bg-slate-50 dark:bg-navy-950 border border-slate-200 dark:border-navy-800 rounded-lg pl-8 pr-3 py-1.5 text-slate-900 dark:text-slate-200 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:border-teal-500"
          />
        </div>
      </div>

      {/* Table */}
      <div className="bg-white dark:bg-navy-900/90 border border-slate-200 dark:border-navy-800 rounded-xl overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 dark:bg-navy-950/80 border-b border-slate-200 dark:border-navy-800 text-slate-500 dark:text-slate-400 font-mono text-[11px] uppercase">
              <tr>
                <th className="py-3 px-4">Event ID</th>
                <th className="py-3 px-4">Timestamp</th>
                <th className="py-3 px-4">User ID</th>
                <th className="py-3 px-4">Item ID</th>
                <th className="py-3 px-4">Action</th>
                <th className="py-3 px-4">Linked Recommendation ID</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-navy-800/60 font-mono text-[11px]">
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
                  <tr key={fb.id || `${fb.user_id}-${fb.item_id}-${fb.timestamp}`} className="hover:bg-slate-50 dark:hover:bg-navy-800/40 transition-colors">
                    <td className="py-3 px-4 text-slate-500 dark:text-slate-400">#{fb.id}</td>
                    <td className="py-3 px-4 text-slate-500 dark:text-slate-400">
                      {fb.timestamp ? new Date(fb.timestamp).toLocaleString() : '—'}
                    </td>
                    <td className="py-3 px-4 text-slate-900 dark:text-slate-200 font-semibold">{fb.user_id}</td>
                    <td className="py-3 px-4 text-teal-600 dark:text-teal-400">{fb.item_id}</td>
                    <td className="py-3 px-4">{getActionBadge(fb.action)}</td>
                    <td className="py-3 px-4 text-slate-500">
                      {fb.recommendation_id ? (
                        <span className="text-slate-700 dark:text-slate-300 font-mono bg-slate-100 dark:bg-navy-950 px-1.5 py-0.5 rounded border border-slate-200 dark:border-navy-800">
                          {fb.recommendation_id}
                        </span>
                      ) : (
                        <span className="italic text-slate-400 dark:text-slate-600">unlinked</span>
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
