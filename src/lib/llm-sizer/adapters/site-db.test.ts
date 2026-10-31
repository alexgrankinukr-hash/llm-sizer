import { expect, it, vi } from 'vitest';
const mocks = vi.hoisted(() => ({ getDb: vi.fn() }));
vi.mock('../../../db/index', () => ({ getDb: mocks.getDb }));
import { upsertNotifyLead } from './site-db';

it('leaves an unsubscribed lead and their tokens untouched on a notify request', async () => {
  const row = { id: 'test-lead', confirmedAt: new Date(), unsubscribedAt: new Date() };
  const db = {
    select: vi.fn(() => ({ from: () => ({ where: () => ({ limit: async () => [row] }) }) })),
    update: vi.fn(), insert: vi.fn(),
  };
  mocks.getDb.mockReturnValue(db);
  expect(await upsertNotifyLead({ email: 'reader@passmail.net', sourceChannel: null, ip: '127.0.0.1' })).toMatchObject({ unsubscribed: true, alreadyConfirmed: false, token: '' });
  expect(db.update).not.toHaveBeenCalled();
  expect(db.insert).not.toHaveBeenCalled();
});
