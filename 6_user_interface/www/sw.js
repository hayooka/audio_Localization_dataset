self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));

self.addEventListener('message', event => {
  if (event.data && event.data.type === 'SHOW_NOTIF') {
    const { title, body, tag } = event.data;
    self.registration.showNotification(title, {
      body,
      tag,
      icon: '/icon-192.png',
      renotify: true,
      requireInteraction: false,
    });
  }
});
