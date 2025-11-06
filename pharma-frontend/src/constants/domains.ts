export type DomainKey = 'Customer/HCP' | 'Account/Institutional' | 'Marketing Campaign' | 'Data/Analytics';

export type DomainConfig = {
  key: DomainKey;
  displayName: string;
  listTypes: string[];
  domainId: number; // Supabase domain_id
};

export const DOMAIN_CONFIGS: DomainConfig[] = [
  {
    key: 'Customer/HCP',
    displayName: 'Customer/HCP Lists',
    listTypes: [
      'Target Lists',
      'Call Lists',
    ],
    domainId: 1
  },
  {
    key: 'Account/Institutional',
    displayName: 'Account/Institutional Lists',
    listTypes: [
      'Formulary Decision-Maker Lists',
      'IDN/Health System Lists',
    ],
    domainId: 2
  },
  {
    key: 'Marketing Campaign',
    displayName: 'Marketing Campaign Lists',
    listTypes: [
      'Event Invitation Lists',
      'Digital Engagement Lists',
    ],
    domainId: 3
  },
  {
    key: 'Data/Analytics',
    displayName: 'Data/Analytics Lists',
    listTypes: [
      'High-Value Prescriber Lists',
      'Competitor Target Lists',
    ],
    domainId: 4
  }
];

// Helper function to get domain config by key
export const getDomainConfig = (key: string): DomainConfig | undefined => {
  return DOMAIN_CONFIGS.find(config => config.key === key);
};

// Helper function to get all domain keys
export const getDomainKeys = (): DomainKey[] => {
  return DOMAIN_CONFIGS.map(config => config.key);
};

// Helper function to get display name for a domain
export const getDomainDisplayName = (key: string): string => {
  const config = getDomainConfig(key);
  return config ? config.displayName : key;
};

// Helper function to get list types for a domain
export const getListTypesForDomain = (key: string): string[] => {
  const config = getDomainConfig(key);
  return config ? config.listTypes : [];
};

// Map old domain names to new ones for backward compatibility
export const DOMAIN_MIGRATION_MAP: Record<string, DomainKey> = {
  'Customer': 'Customer/HCP',
  'Account': 'Account/Institutional',
  'Marketing': 'Marketing Campaign',
  'Data': 'Data/Analytics'
};

// Helper function to migrate old domain names
export const migrateDomainName = (oldDomain: string): DomainKey => {
  return DOMAIN_MIGRATION_MAP[oldDomain] || oldDomain as DomainKey;
};
