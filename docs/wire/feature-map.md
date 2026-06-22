# pplx — feature ⇄ endpoint map (2026-06-22T16-24-27Z)

How the live web frontend's components/hooks map to API endpoints. Derived from
the Vite chunk symbols each endpoint literal appears in (see
`scripts/enumerate-endpoints.py`), NOT from observed runtime traffic — CDP capture
is blocked by Cloudflare (see `endpoint-gap-analysis.md`). So this is a *code-level*
map: the React component/hook name is a strong but not authoritative feature label.

726 endpoints total; 53 descriptive symbols own ≥2 endpoints each.

| Component / hook (feature) | # | endpoints |
|---|---|---|
| `FinancePortfolioPage` | 31 | `/rest/finance/documents/{market_identifier}`, `/rest/finance/earnings/{market_identifier}`, `/rest/finance/financials/v2/{market_identifier}`, `/rest/finance/financials/v3/{market_identifier}`, `/rest/finance/financials/{market_identifier}`, `/rest/finance/market-status/{market_identifier}`, `/rest/finance/personal/liabilities`, `/rest/finance/personal/overview` … |
| `useUpdateArticleSectionImage` | 13 | `/rest/article/reorder/{article_slug}`, `/rest/article/section/add/media_item/{section_uuid}`, `/rest/article/section/hero_offset/{section_uuid}`, `/rest/article/section/image/{section_uuid}`, `/rest/article/section/media/remove/{section_uuid}`, `/rest/article/section/title/{section_uuid}`, `/rest/article/section/upload_image/{section_uuid}`, `/rest/article/section/{section_uuid}` … |
| `SpaceSharingPanel` | 12 | `/rest/collections/invitations`, `/rest/collections/invite_to_collection`, `/rest/collections/respond_to_invitation/{collection_uuid}`, `/rest/collections/{collection_uuid}/access`, `/rest/collections/{collection_uuid}/join-request/approve`, `/rest/collections/{collection_uuid}/join-request/cancel`, `/rest/collections/{collection_uuid}/join-request/deny`, `/rest/collections/{collection_uuid}/join-requests` … |
| `useAssetPublishControl` | 10 | `/rest/assets/sites/validate-subdomain/{subdomain}`, `/rest/assets/sites/{site_id}/publish-info`, `/rest/assets/{asset_id}/access`, `/rest/assets/{asset_id}/invite`, `/rest/assets/{asset_id}/members`, `/rest/assets/{asset_id}/published-access`, `/rest/assets/{asset_id}/published-invite`, `/rest/assets/{asset_id}/published-members` … |
| `HealthPageWrapper` | 7 | `/rest/health-assistant/lab-results/upload`, `/rest/health-assistant/lab-results/upload/{upload_id}`, `/rest/health-assistant/lab-results/upload/{upload_id}/status`, `/rest/health-assistant/lab-results/uploads`, `/rest/health-feedback/`, `/rest/health-feedback/waitlist`, `/rest/sse/health_summary` |
| `healthMutations` | 7 | `/rest/connectors/medical_records/connect`, `/rest/connectors/medical_records/terms-of-service`, `/rest/connectors/medical_records/user-profile`, `/rest/connectors/wearables/connect`, `/rest/health-assistant/landing-page/onboarding-status`, `/rest/health-assistant/provider-connections/{health_provider_type}/{connection_id}/connect`, `/rest/health-assistant/provider-connections/{health_provider_type}/{connection_id}/disconnect` |
| `useOrganizationMembers` | 7 | `/rest/enterprise/organization/invite/{org_invite_uuid}`, `/rest/enterprise/organization/invite/{org_invite_uuid}/resend`, `/rest/enterprise/organization/user/{org_user_uuid}`, `/rest/enterprise/organization/user/{org_user_uuid}/make-admin`, `/rest/enterprise/organization/user/{org_user_uuid}/make-member`, `/rest/enterprise/trial/update/{org_user_uuid}/{update_type}`, `/rest/enterprise/user/organization/members` |
| `OrgInviteModal` | 6 | `/api/user`, `/rest/enterprise/organization/invite/{org_invite_uuid}`, `/rest/enterprise/organization/invite/{org_invite_uuid}/accept`, `/rest/enterprise/organization/invite/{org_invite_uuid}/decline`, `/rest/enterprise/organization/shareable-invite/join/{open_invite_link_uuid}`, `/rest/enterprise/organization/shareable-invite/{open_invite_link_uuid}` |
| `ArtifactsPage` | 5 | `/rest/assets/`, `/rest/assets/shared-with-me`, `/rest/assets/shared-with-me/{asset_id}`, `/rest/assets/signed-url/{asset_id}`, `/rest/assets/{asset_id}` |
| `RestaurantBookingModal` | 5 | `/rest/travel/restaurants/info`, `/rest/travel/restaurants/reservations`, `/rest/travel/restaurants/search`, `/rest/travel/restaurants/slots/delete`, `/rest/travel/restaurants/slots/lock` |
| `SubscriptionDetailsModal` | 5 | `/rest/enterprise/cancel-subscription`, `/rest/stripe/cancel-subscription`, `/rest/stripe/cancellation-config`, `/rest/stripe/claim-retention-offer`, `/rest/stripe/customer-invoices` |
| `useComputerMemoryInteractive` | 5 | `/rest/memories/delete`, `/rest/sse/computer/memory/dream-instructions`, `/rest/sse/computer/memory/dream-settings`, `/rest/sse/computer/memory/dream-settings/enabled-flags`, `/rest/sse/computer/memory/wiki-pages/delete` |
| `useSpaceSlackBindingsQuery` | 5 | `/rest/spaces/{space_uuid}/slack-bindings`, `/rest/spaces/{space_uuid}/slack-bindings/bind`, `/rest/spaces/{space_uuid}/slack-bindings/channels`, `/rest/spaces/{space_uuid}/slack-bindings/confirm`, `/rest/spaces/{space_uuid}/slack-bindings/unbind` |
| `ComputerProviders` | 4 | `/rest/billing/account-promo-message`, `/rest/billing/credits/check`, `/rest/connector-service/warm-cache`, `/rest/sse/entry_creation_events` |
| `CopyConnectorLinkButton` | 4 | `/rest/connections/sources/{source_id}/tool_permissions`, `/rest/connections/sources/{source_id}/tool_permissions/by_classification`, `/rest/connector-service/connectors/{source_id}/settings-config`, `/rest/connector-service/connectors/{source_id}/tools` |
| `GenericWorkflowModal` | 4 | `/rest/triggers/event-schemas`, `/rest/triggers/event-subscriptions`, `/rest/triggers/event-subscriptions/${e}`, `/rest/triggers/event-subscriptions/{trigger_id}` |
| `HotelRoomBookingPage` | 4 | `/rest/stripe/get-customer-billing-info`, `/rest/travel/hotels/rate-verification`, `/rest/travel/hotels/reservations`, `/rest/travel/hotels/user/discount-offer` |
| `RequestAccessModal` | 4 | `/rest/assets/{asset_id}/request-access`, `/rest/assets/{asset_id}/request-access-info`, `/rest/thread/request-access-info/{entry_uuid_or_slug}`, `/rest/thread/request-access/{entry_uuid_or_slug}` |
| `useConnectorSharedSettings` | 4 | `/rest/collections/{collection_uuid}/connector_settings`, `/rest/collections/{collection_uuid}/connector_settings/connections`, `/rest/collections/{collection_uuid}/connector_settings/connections/sources/{source_id}`, `/rest/collections/{collection_uuid}/connector_settings/sources/{source_id}` |
| `useGetSubscriptionBillingInfo` | 4 | `/rest/enterprise/organization/subscription-info`, `/rest/enterprise/resubscribe`, `/rest/stripe/resubscribe`, `/rest/stripe/subscription-billing-info` |
| `useHealthAttributes` | 4 | `/rest/memories/delete-health-attributes`, `/rest/memories/get-health-attributes`, `/rest/memories/get-health-attributes-config`, `/rest/memories/write-health-attributes` |
| `useShoppingTryOn` | 4 | `/rest/shopping/generate-shopping-try-on`, `/rest/shopping/try-on-photo/delete`, `/rest/shopping/try-on-photo/generate-avatar`, `/rest/shopping/try-on-photo/save` |
| `useVezgoConnect` | 4 | `/rest/finance/personal`, `/rest/finance/personal/plaid/connections`, `/rest/finance/personal/vezgo/accounts/register`, `/rest/finance/personal/vezgo/connect-url` |
| `BraintreeSubscriptionModal` | 3 | `/rest/billing/braintree/cancel-subscription`, `/rest/billing/braintree/renew-subscription`, `/rest/billing/braintree/subscription-details` |
| `ThreadShareButton` | 3 | `/rest/thread/invite_to_thread`, `/rest/thread/remove_thread_member`, `/rest/thread/{entry_uuid_or_slug}/share_slug` |
| `useOrgBilling` | 3 | `/rest/enterprise/checkout-session`, `/rest/enterprise/customer-portal`, `/rest/enterprise/subscription` |
| `useOrgPremiumSecureFeatures` | 3 | `/rest/enterprise/organization/premium-secure-billing-update`, `/rest/enterprise/organization/premium-secure-eligibility`, `/rest/enterprise/organization/subscription-info` |
| `useOrgSourceAuth` | 3 | `/rest/connections/connect`, `/rest/enterprise/user/organization/sources/{source_id}/disconnect`, `/rest/enterprise/user/organization/sources/{source_id}/permission` |
| `usePriceAlertMutations` | 3 | `/rest/autosuggest/finance/tasks-autosuggest`, `/rest/tasks/finance`, `/rest/tasks/finance/{task_id}` |
| `AddSharedLimitGroupsModal` | 2 | `/rest/organizations/{organization_id}/credit-limits/{credit_limit_id}/available-groups`, `/rest/organizations/{organization_id}/credit-limits/{credit_limit_id}/groups:batchAdd` |
| `AddSharedLimitMembersModal` | 2 | `/rest/organizations/{organization_id}/credit-limits/{credit_limit_id}/available-users`, `/rest/organizations/{organization_id}/credit-limits/{credit_limit_id}/users:batchAdd` |
| `AutoRefillForm` | 2 | `/rest/billing/credits/spending-limit`, `/rest/billing/credits/topup-context` |
| `CalendarSelectionModal` | 2 | `/rest/email_assistant/calendars`, `/rest/email_assistant/calendars/selected` |
| `ConnectorManageSetupSection` | 2 | `/rest/sources/{source_id}/setup-config`, `/rest/sources/{source_id}/setup-config/{owner_scope}/{setup_config_id}` |
| `JobsMode` | 2 | `/rest/jobs/filters`, `/rest/jobs/search` |
| `NotificationSection` | 2 | `/rest/notifications/preferences`, `/rest/notifications/unsubscribe/{notification_type}` |
| `PaymentModal` | 2 | `/rest/billing/get-coupon-metadata`, `/rest/stripe/update-checkout-interval` |
| `SetSharedLimitModal` | 2 | `/rest/organizations/{organization_id}/credit-limits`, `/rest/organizations/{organization_id}/credit-limits/{credit_limit_id}` |
| `ShoppingMode` | 2 | `/rest/shopping/search-shopping-for-entry`, `/rest/sse/shopping_review_summary` |
| `SubscriptionTransferModal` | 2 | `/rest/billing/subscription-transfer/execute`, `/rest/billing/subscription-transfer/initiate` |
| `SwitchIntervalModal` | 2 | `/rest/stripe/switch-billing-interval`, `/rest/stripe/switch-billing-interval-preview` |
| `TOTPSetupModal` | 2 | `/api/auth/totp/enroll`, `/api/auth/totp/verify` |
| `UpgradeModal` | 2 | `/rest/stripe/upgrade-subscription`, `/rest/stripe/upgrade-subscription-preview` |
| `WatchlistModal` | 2 | `/rest/homepage-widgets/watchlist/categories`, `/rest/homepage-widgets/watchlist/subscription` |
| `miscMutations` | 2 | `/rest/entry/convert-to-report/{entry_uuid}`, `/rest/user/delete-account` |
| `remoteMcpMutations` | 2 | `/rest/sources/custom`, `/rest/sources/custom/{source_id}` |
| `useDownloadableAsset` | 2 | `/rest/connectors/save-file`, `/rest/deeper-research/export-asset` |
| `useGetDowngradeSubscriptionPreview` | 2 | `/rest/stripe/downgrade-subscription`, `/rest/stripe/downgrade-subscription-preview` |
| `useGetOrgSuggestions` | 2 | `/rest/enterprise/organization/join-request`, `/rest/enterprise/v2/user/org-suggestions` |
| `useNtpUpsellsQuery` | 2 | `/rest/ntp/upsell/`, `/rest/ntp/upsell/interacted` |
| `usePinnedWorkflows` | 2 | `/rest/workflows/pins`, `/rest/workflows/pins/{workflow_id}` |
| `useReplaceSkillFile` | 2 | `/rest/skills`, `/rest/skills/${e}` |
| `useSpaceTeamsBindingsQuery` | 2 | `/rest/spaces/{space_uuid}/ms-teams-bindings`, `/rest/spaces/{space_uuid}/ms-teams-bindings/unbind` |
