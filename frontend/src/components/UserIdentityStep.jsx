import React, { useState, useEffect } from 'react';
import { UserCheck, Snowflake, Flame, RefreshCw, AlertCircle, History, Clock, FileSpreadsheet, ChevronDown, ChevronUp } from 'lucide-react';
import { api } from '../services/api';

/**
 * UserIdentityStep Component
 * Provides:
 * 1. User ID validation (non-empty, safe chars, max length)
 * 2. Real-time Cold Start vs Warm Start detection
 * 3. Badge display (Cold Start vs Warm Start)
 * 4. Collapsible Persistent Interaction History drawer
 * 5. Informative explanation for cold-start users
 */
export default function UserIdentityStep({
  clientId,
  userId,
  onUserIdChange,
  onStatusChange,
  showHistoryDrawer = true,
  className = ''
}) {
  const [statusData, setStatusData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [validationError, setValidationError] = useState('');
  const [historyList, setHistoryList] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [showHistory, setShowHistory] = useState(false);

  // Validate User ID format
  const validateInput = (id) => {
    if (!id || !id.trim()) {
      return 'User ID is required. Please enter a valid identifier.';
    }
    const clean = id.trim();
    if (clean.length > 128) {
      return 'User ID cannot exceed 128 characters.';
    }
    if (clean.includes('..') || clean.includes('/') || clean.includes('\\')) {
      return 'User ID cannot contain path characters (/, \\, ..).';
    }
    return '';
  };

  // Fetch Cold / Warm Status whenever userId or clientId changes
  const checkStatus = async (uidToTest) => {
    const targetUid = uidToTest !== undefined ? uidToTest : userId;
    const err = validateInput(targetUid);
    setValidationError(err);

    if (err || !targetUid || !clientId) {
      setStatusData(null);
      if (onStatusChange) onStatusChange(null);
      return;
    }

    setLoading(true);
    try {
      const res = await api.getUserHistoryStatus(clientId, targetUid.trim());
      setStatusData(res);
      if (onStatusChange) onStatusChange(res);

      // If warm start, preload history summary
      if (res.status === 'warm_start' && res.history_count > 0) {
        fetchHistory(targetUid.trim());
      } else {
        setHistoryList([]);
      }
    } catch (e) {
      console.warn('Failed to retrieve user status:', e);
      setStatusData(null);
      if (onStatusChange) onStatusChange(null);
    } finally {
      setLoading(false);
    }
  };

  const fetchHistory = async (uid) => {
    setHistoryLoading(true);
    try {
      const res = await api.getUserHistory(clientId, uid || userId.trim(), 15);
      setHistoryList(res.history || []);
    } catch (e) {
      console.warn('Failed to fetch user history records:', e);
    } finally {
      setHistoryLoading(false);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      if (userId && userId.trim()) {
        checkStatus(userId.trim());
      } else {
        setStatusData(null);
        setValidationError('User ID is required. Please enter an identifier.');
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [userId, clientId]);

  const isCold = statusData?.status === 'cold_start';
  const isWarm = statusData?.status === 'warm_start';

  return (
    <div className={`p-4 rounded-2xl dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 shadow-sm transition-all ${className}`}>
      {/* Header & Badges */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-xl bg-teal-500/10 text-teal-600 dark:text-teal-400 flex items-center justify-center font-bold">
            <UserCheck className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider dark:text-coolwhite text-slate-800">
              Active User Identification & Profile
            </h3>
            <p className="text-[11px] text-muted">
              Scope: <span className="font-mono font-semibold text-teal-600 dark:text-teal-400">{clientId}</span>
            </p>
          </div>
        </div>

        {/* Cold vs Warm Start Badge */}
        <div className="flex items-center gap-2">
          {loading ? (
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-slate-100 dark:bg-navy-950 text-muted border dark:border-navy-800 border-slate-200">
              <RefreshCw className="w-3 h-3 animate-spin text-teal-500" />
              Checking profile...
            </span>
          ) : isCold ? (
            <span
              data-testid="user-status-cold"
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-sky-500/15 text-sky-600 dark:text-sky-400 border border-sky-500/30"
              title="Cold Start: No prior saved interactions found in this tenant."
            >
              <Snowflake className="w-3.5 h-3.5" />
              <span>Cold Start</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-sky-500/20 font-mono">0 history</span>
            </span>
          ) : isWarm ? (
            <span
              data-testid="user-status-warm"
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30"
              title="Warm Start: User has persistent interaction history in this tenant."
            >
              <Flame className="w-3.5 h-3.5" />
              <span>Warm Start</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-emerald-500/20 font-mono">
                {statusData?.history_count} {statusData?.history_count === 1 ? 'event' : 'events'}
              </span>
            </span>
          ) : null}
        </div>
      </div>

      {/* Input Field & Inline Validation */}
      <div className="space-y-1.5">
        <label className="block text-xs font-semibold dark:text-coolwhite text-slate-700">
          User Identifier (User ID / Customer ID)
        </label>
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={userId}
            onChange={(e) => onUserIdChange(e.target.value)}
            placeholder="Enter User ID (e.g. 9272, user_1, cold_user_42)"
            className={`w-full dark:bg-navy-950 bg-slate-50 border rounded-xl px-3.5 py-2 text-xs font-mono dark:text-coolwhite text-slate-900 focus:outline-none transition-all ${
              validationError
                ? 'border-red-500 focus:border-red-600'
                : 'dark:border-navy-800 border-slate-200 focus:border-teal-500'
            }`}
          />
          <button
            type="button"
            onClick={() => checkStatus()}
            disabled={loading}
            className="shrink-0 px-3 py-2 rounded-xl border dark:border-navy-800 border-slate-200 hover:border-teal-500/40 text-xs font-semibold text-muted hover:dark:text-coolwhite hover:text-slate-900 transition-all cursor-pointer flex items-center gap-1"
            title="Refresh User Profile Status"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Verify</span>
          </button>
        </div>

        {/* Validation Error Banner */}
        {validationError && (
          <div className="flex items-center gap-1.5 text-xs text-red-500 pt-1">
            <AlertCircle className="w-3.5 h-3.5 shrink-0" />
            <span>{validationError}</span>
          </div>
        )}
      </div>

      {/* Cold Start Information Callout */}
      {isCold && !validationError && (
        <div className="mt-3 p-3 rounded-xl dark:bg-sky-500/10 bg-sky-50 border border-sky-500/20 text-xs text-sky-800 dark:text-sky-300 flex items-start gap-2.5">
          <Snowflake className="w-4 h-4 shrink-0 text-sky-500 mt-0.5" />
          <div className="space-y-0.5">
            <div className="font-semibold">Cold-Start Profile Detected</div>
            <p className="text-[11px] leading-relaxed text-sky-700 dark:text-sky-400">
              This user has 0 saved interactions for tenant <strong className="font-mono">{clientId}</strong>. Recommendations are generated using <strong>dataset properties</strong>, column distributions, and general heuristics. As the user views or interacts with charts, their history is persistently recorded to transition them to <strong>Warm Start</strong>.
            </p>
          </div>
        </div>
      )}

      {/* Warm Start Information & History Drawer Toggle */}
      {isWarm && !validationError && showHistoryDrawer && (
        <div className="mt-3 pt-3 border-t dark:border-navy-800 border-slate-100">
          <div className="flex items-center justify-between">
            <div className="text-xs text-muted flex items-center gap-2">
              <Clock className="w-3.5 h-3.5 text-emerald-500" />
              <span>
                Last active:{' '}
                <strong className="dark:text-coolwhite text-slate-800">
                  {statusData?.last_active ? new Date(statusData.last_active).toLocaleString() : 'Recently'}
                </strong>
              </span>
            </div>

            <button
              type="button"
              onClick={() => setShowHistory(!showHistory)}
              className="text-xs font-semibold text-teal-600 dark:text-teal-400 hover:underline flex items-center gap-1 cursor-pointer"
            >
              <History className="w-3.5 h-3.5" />
              <span>{showHistory ? 'Hide User History' : `View User History (${statusData?.history_count})`}</span>
              {showHistory ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
          </div>

          {/* Collapsible History Drawer */}
          {showHistory && (
            <div className="mt-3 p-3 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 space-y-2 max-h-64 overflow-y-auto">
              <div className="flex items-center justify-between text-[11px] font-bold text-muted uppercase">
                <span>Recent Interaction Records</span>
                <span>Tenant: {clientId}</span>
              </div>

              {historyLoading ? (
                <div className="text-center py-4 text-xs text-muted flex items-center justify-center gap-2">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin text-teal-500" />
                  Loading history log...
                </div>
              ) : historyList.length === 0 ? (
                <div className="text-center py-3 text-xs text-muted">
                  No interactions found in local persistence.
                </div>
              ) : (
                <div className="space-y-2">
                  {historyList.map((entry) => (
                    <div
                      key={entry.event_id}
                      className="p-2.5 rounded-lg dark:bg-navy-900 bg-white border dark:border-navy-800/80 border-slate-200 text-xs space-y-1"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-teal-600 dark:text-teal-400 font-mono text-[11px]">
                          {entry.entry_type}
                        </span>
                        <span className="text-[10px] text-muted">
                          {entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString() : ''}
                        </span>
                      </div>

                      {entry.dataset_id && (
                        <div className="flex items-center gap-1 text-[11px] text-muted">
                          <FileSpreadsheet className="w-3 h-3 text-purple-400" />
                          <span>Dataset: {entry.dataset_id}</span>
                        </div>
                      )}

                      {entry.items_or_charts && entry.items_or_charts.length > 0 && (
                        <div className="text-[10px] text-muted pt-0.5 truncate">
                          Items: {entry.items_or_charts.map(c => c.chart_type || c.item_id || c.title).filter(Boolean).join(', ')}
                        </div>
                      )}

                      {entry.feedback && (
                        <div className="text-[10px] text-emerald-500 font-semibold">
                          Feedback: {entry.feedback}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
