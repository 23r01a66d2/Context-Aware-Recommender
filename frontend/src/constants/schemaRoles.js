/**
 * Centralized Canonical Schema Roles & Data Types
 * Matches the backend SchemaRole enum in src/schema_mapper.py exactly.
 */

export const CANONICAL_SCHEMA_ROLES = [
  {
    value: 'USER_ID',
    label: 'USER_ID (Customer Identifier)',
    color: 'text-indigo-400',
    description: 'Primary user/customer identifier for history aggregation',
  },
  {
    value: 'ITEM_ID',
    label: 'ITEM_ID (Product Identifier)',
    color: 'text-emerald-400',
    description: 'Candidate item/product identifier for candidate pools',
  },
  {
    value: 'TIMESTAMP',
    label: 'TIMESTAMP (Chronological Event Time)',
    color: 'text-sky-400',
    description: 'Interaction timestamp for chronological splitting & leakage prevention',
  },
  {
    value: 'TARGET',
    label: 'TARGET (Binary / Conversion Signal)',
    color: 'text-amber-400',
    description: 'Supervised recommendation training objective',
  },
  {
    value: 'BEHAVIOR_FEATURE',
    label: 'BEHAVIOR_FEATURE (Point-in-Time History)',
    color: 'text-blue-400',
    description: 'User engagement metrics & behavioral features',
  },
  {
    value: 'CONTENT_FEATURE',
    label: 'CONTENT_FEATURE (Item Metadata)',
    color: 'text-teal-400',
    description: 'Candidate product attributes (category, price, brand)',
  },
  {
    value: 'CONTEXT_FEATURE',
    label: 'CONTEXT_FEATURE (Session / Device / Env)',
    color: 'text-purple-400',
    description: 'Session context (device, channel, season, location)',
  },
  {
    value: 'IGNORE',
    label: 'IGNORE (Exclude / Leakage Prevention)',
    color: 'text-slate-500',
    description: 'Post-decision outcomes or unused columns (strictly ignored)',
  },
];

export const CANONICAL_DATA_TYPES = [
  'categorical',
  'numerical',
  'timestamp',
  'id',
  'text',
];
