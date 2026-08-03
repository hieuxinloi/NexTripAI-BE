# Cấu hình Google Routes API

Tài liệu này hướng dẫn tạo credential để NexTripAI lấy **khoảng cách đường bộ và thời gian tuyến thực tế**.

## 1. Bật Routes API

- Mở [Routes API trong Google Cloud Console](https://console.cloud.google.com/apis/library/routes.googleapis.com).
- Chọn đúng Google Cloud project.
- Bấm **Enable**.
- Project cần có billing account. Nếu đủ điều kiện, có thể dùng credit dùng thử của Google Cloud.

Tài liệu chính thức: [Set up the Routes API](https://developers.google.com/maps/documentation/routes/get-api-key).

## 2. Tạo API key

- Mở [Credentials](https://console.cloud.google.com/apis/credentials).
- Chọn **Create credentials → API key**.
- Sao chép key bắt đầu bằng `AIza...`.
- Trong phần **API restrictions**, chọn **Restrict key** và chỉ cho phép **Routes API**.

Không dùng OAuth Client ID hoặc service-account JSON cho biến này.

## 3. Cấu hình Backend

Trong `NexTripAI-BE/.env`:

```env
GOOGLE_MAPS_API_KEY=AIzaSy...
ROUTES_TIMEOUT_SECONDS=10
ROUTES_CACHE_TTL_SECONDS=604800
```

Không commit `.env` hoặc gửi API key lên GitHub. Backend mới là nơi gọi Routes API; key không được đưa vào Frontend.

## 4. Quota và cách tính

Xem bảng giá tại [Google Maps Platform Pricing](https://developers.google.com/maps/billing-and-pricing/pricing).

- Compute Routes Essentials: free cap hiện tại 10.000 request/tháng.
- Compute Route Matrix Essentials: free cap hiện tại 10.000 elements/tháng.
- Các request Enterprise, trong đó có thể bao gồm `TWO_WHEELER`, có free cap riêng 1.000 request/elements.
- Route Matrix tính theo `số origin × số destination`, không chỉ theo số request HTTP.

Ví dụ: 1 điểm xuất phát và 5 điểm đến = 5 route-matrix elements.

Giá và quota có thể thay đổi; luôn kiểm tra bảng giá chính thức trước khi triển khai production.
