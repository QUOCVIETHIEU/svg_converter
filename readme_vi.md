# Nhóm I/O & chế độ
	--input: đường dẫn ảnh nguồn PNG/JPEG.
	--output: file SVG xuất ra.
	--mode {auto,color,mono}
		•	auto: tự quyết số lớp màu (thường dựa heuristic).
		•	color: ép dùng đúng k màu. Hợp với logo/phẳng.
		•	mono: đen–trắng (threshold) → cho contour “sắc” nhất, dễ mài răng cưa.

	--k: Số cụm màu trong file gốc
		•	Tăng: giữ nhiều sắc độ/chi tiết → nhiều path hơn, nặng file, dễ răng cưa nhẹ ở vùng biên lẫn màu.
		•	Giảm: đơn giản hoá màu → mượt thị giác, ít path, hợp logo.
		•	Gợi ý: logo 3–12; icon/flat 4–8; ảnh posterize 8–16; 50 là cao (chỉ khi cần chi tiết).

# Nhóm vùng ảnh / tiền xử lý
	--min-area INT (mặc định 80 px²): Bỏ qua mảng nhỏ hơn ngưỡng.
		•	Tăng: loại hạt bụi/nhỏ lẻ → SVG sạch & mượt hơn.
		•	Giảm: giữ chi tiết li ti → dễ lởm chởm.
		•	Logo: 50–300; scan nhiễu: 150–800.

	--area-max-ratio FLOAT (mặc định 0.92): Giới hạn tỷ lệ diện tích một vùng so với toàn ảnh
		•	Tăng gần 1.0: dễ giữ mảng lớn (hậu cảnh), có thể làm thuật toán coi cả nền là một khối → khó mượt.
		•	Giảm: mạnh tay coi mảng quá lớn là nền → contour tập trung vào đối tượng.

	--super-sample INT (SSAA, mặc định 4): Phóng to ảnh nội bộ trước khi dò biên rồi thu về (anti-alias)
		•	Tăng: biên mịn thấy rõ (đặc biệt đường cong) nhưng tốn CPU/RAM.
		•	Giảm: nhanh hơn, biên “bậc thang” hơn.
		•	Logo: 3–6; ảnh lớn: 2–4.
	
	--decimate INT (mặc định 1): Lấy mẫu thưa bớt điểm contour đầu vào.
		•	Tăng: ít điểm → path mượt hơn nhưng có nguy cơ mất chi tiết/gãy góc.
		•	Giảm (về 1): giữ đủ điểm → mịn + đúng hình nhất, nhưng cần các bước mượt khác để tránh răng cưa.
	
	--no-auto-crop (flag): Tắt tự động cắt sát đối tượng. Auto-crop giúp tăng độ mượt (ít vùng rìa/nhiễu)
	--crop x,y,w,h: Cắt thủ công. Dùng khi muốn tập trung vào vùng quan trọng để vector hoá mượt hơn.
	--subpixel-sigma FLOAT (mặc định 0.8): Gaussian nhẹ trước khi lấy biên (mịn cận subpixel).
		•	Tăng: bớt răng cưa, nhưng có thể bo tròn góc.
		•	Giảm: sắc góc, dễ răng cưa; 0 = tắt.
		•	Logo: 0.5–1.0; mono/scan nhiễu: 0.8–1.5.

# Nhóm “sửa chữa biên” (repair) & hình thái học
	--no-repair (flag): Tắt bước hàn mép/điền lỗ nhỏ sau segment hoá. Tắt sẽ “trung thực” nhưng dễ lỗ châm kim, răng cưa.
	
	--repair-clean FLOAT (mặc định 0.01, theo kích thước ảnh): Ngưỡng hợp nhất/điền các khe nhỏ khi repair.
		•	Tăng: mạnh tay làm sạch → mượt hơn, nhưng có thể dính mảng mỏng.
		•	Giảm: ít can thiệp → nhiều khe nhỏ còn lại.
	
	--morph-close INT (mặc định 4, px): Closing (dilate rồi erode) lấp các khe đứt.
		•	Tăng: bo mép, liền nét (mượt) nhưng có thể làm phình chi tiết nhỏ.
		•	Giảm: ít lấp khe → giữ nét sắc, có thể răng cưa.
	
	--morph-open INT (mặc định 1, px): Opening (erode rồi dilate) loại nhiễu hạt nhỏ.
		•	Tăng: sạch hơn, nhưng mất chi tiết mảnh.
		•	Giảm: giữ chi tiết, dễ lạo xạo.

# Nhóm bắt thẳng/góc & sai số khớp (rất quan trọng cho độ mượt thị giác)
	--corner-eps FLOAT (mặc định 0.9): Độ “nhạy” phát hiện góc.
		•	Tăng: ít điểm góc → đường được coi là cong nhiều hơn → mượt hơn, nhưng có thể bo tròn góc đáng lẽ phải nhọn.
		•	Giảm: nhiều điểm góc → góc sắc chính xác, nhưng có cảm giác “gãy khúc”.
	
	--line-tol FLOAT (mặc định 0.1): Dung sai để nhận ra một đoạn là đường thẳng (thay vì cong).
		•	Tăng: dễ “snap” thành thẳng → biên phẳng, mượt thị giác ở logo có cạnh thẳng.
		•	Giảm: ít đoạn thẳng → nhiều đoạn cong; mượt cong nhưng cạnh thẳng có thể lượn nhẹ.
	
	--max-err FLOAT (mặc định 0.26): Sai số khớp tối đa (fit error) khi xấp xỉ bằng đoạn thẳng/curve.
		•	Tăng: bớt điểm điều khiển → path gọn, mượt tổng thể, nhưng có thể lệch hình nhỏ.
		•	Giảm: bám hình sát → nhiều điểm điều khiển → có nguy cơ “răng cưa” thị giác.

# Nhóm làm mượt cong chỉ trong mono
	--curve-sigma FLOAT (mặc định 1.4; 0 = tắt): Gaussian chỉ áp dụng trên đoạn cong (không ảnh hưởng cạnh thẳng).
		•	Tăng: cong mượt rõ rệt, đặc biệt mono, hạn chế bậc thang. Quá cao có thể “lụn” chi tiết.
		•	Giảm: giữ chi tiết cong sắc hơn.