import time, os, sys, gc
import math
import cv_lite
import ulab.numpy as np
from media.sensor import *
from media.display import *
from media.media import *
from ybUtils.YbUart import YbUart
from libs.YbProtocol import YbProtocol
class KalmanFilter2D:
	def __init__(self, process_noise=1e-2, measurement_noise=1e-1):
		self.dt = 1.0
		self.state = [0.0, 0.0, 0.0, 0.0]
		self.P = [[1000.0, 0.0, 0.0, 0.0],
				  [0.0, 1000.0, 0.0, 0.0],
				  [0.0, 0.0, 1000.0, 0.0],
				  [0.0, 0.0, 0.0, 1000.0]]
		q = process_noise
		self.Q = [[q, 0, 0, 0],
				  [0, q, 0, 0],
				  [0, 0, q, 0],
				  [0, 0, 0, q]]
		self.H = [[1, 0, 0, 0],
				  [0, 1, 0, 0]]
		r = measurement_noise
		self.R = [[r, 0],
				  [0, r]]
		self.initialized = False
	def predict(self):
		if not self.initialized:
			return
		F = [[1, 0, self.dt, 0],
			 [0, 1, 0, self.dt],
			 [0, 0, 1, 0],
			 [0, 0, 0, 1]]
		s = self.state
		self.state = [F[0][0]*s[0] + F[0][1]*s[1] + F[0][2]*s[2] + F[0][3]*s[3],
					  F[1][0]*s[0] + F[1][1]*s[1] + F[1][2]*s[2] + F[1][3]*s[3],
					  F[2][0]*s[0] + F[2][1]*s[1] + F[2][2]*s[2] + F[2][3]*s[3],
					  F[3][0]*s[0] + F[3][1]*s[1] + F[3][2]*s[2] + F[3][3]*s[3]]
		P = self.P
		FP = [[sum(F[i][k]*P[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
		FPFT = [[sum(FP[i][k]*F[j][k] for k in range(4)) for j in range(4)] for i in range(4)]
		self.P = [[FPFT[i][j] + self.Q[i][j] for j in range(4)] for i in range(4)]
	def update(self, z):
		x, y = z
		if not self.initialized:
			self.state = [x, y, 0.0, 0.0]
			self.initialized = True
			return
		hx = [self.state[0], self.state[1]]
		residual = [x - hx[0], y - hx[1]]
		HP = [[sum(self.H[i][k]*self.P[k][j] for k in range(4)) for j in range(4)] for i in range(2)]
		HPHT = [[sum(HP[i][k]*self.H[j][k] for k in range(4)) for j in range(2)] for i in range(2)]
		S = [[HPHT[i][j] + self.R[i][j] for j in range(2)] for i in range(2)]
		det = S[0][0]*S[1][1] - S[0][1]*S[1][0]
		if abs(det) < 1e-10:
			return
		S_inv = [[S[1][1]/det, -S[0][1]/det],
				 [-S[1][0]/det, S[0][0]/det]]
		PHT = [[sum(self.P[i][k]*self.H[j][k] for k in range(2)) for j in range(2)] for i in range(4)]
		K = [[sum(PHT[i][k]*S_inv[k][j] for k in range(2)) for j in range(2)] for i in range(4)]
		self.state = [self.state[i] + sum(K[i][j]*residual[j] for j in range(2)) for i in range(4)]
		I = [[1 if i==j else 0 for j in range(4)] for i in range(4)]
		KH = [[sum(K[i][k]*self.H[k][j] for k in range(2)) for j in range(4)] for i in range(4)]
		I_KH = [[I[i][j] - KH[i][j] for j in range(4)] for i in range(4)]
		self.P = [[sum(I_KH[i][k]*self.P[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
	def get_position(self):
		return (int(round(self.state[0])), int(round(self.state[1]))) if self.initialized else None
	def reset(self):
		self.initialized = False
		self.state = [0.0]*4
		self.P = [[1000.0, 0, 0, 0], [0, 1000.0, 0, 0], [0, 0, 1000.0, 0], [0, 0, 0, 1000.0]]
	def get_error(self):
		if not self.initialized:
			return 0.0, 0.0, 0.0, 0.0
		pos_err_x = math.sqrt(max(0, self.P[0][0]))
		pos_err_y = math.sqrt(max(0, self.P[1][1]))
		vel_err_x = math.sqrt(max(0, self.P[2][2]))
		vel_err_y = math.sqrt(max(0, self.P[3][3]))
		return pos_err_x, pos_err_y, vel_err_x, vel_err_y
pto = YbProtocol()
uart1 = YbUart(baudrate=115200)
uart1.send("Hello kankanworld\n\r")
DETECT_WIDTH = 640
DETECT_HEIGHT = 480
IMAGE_CENTER_X = DETECT_WIDTH // 2
lcd_width = DETECT_WIDTH
lcd_height = DETECT_HEIGHT
DISPLAY_MODE = "LCD"
if DISPLAY_MODE == "VIRT":
	DISPLAY_WIDTH = DETECT_WIDTH
	DISPLAY_HEIGHT = DETECT_HEIGHT
elif DISPLAY_MODE == "LCD":
	DISPLAY_WIDTH = 640
	DISPLAY_HEIGHT = 480
else:
	raise ValueError("Unknown DISPLAY_MODE")
sensor = None
sensor = Sensor(width=1920, height=1080)
sensor.reset()
sensor.set_framesize(width=320, height=240)
sensor.set_pixformat(Sensor.RGB565)
if DISPLAY_MODE == "VIRT":
	Display.init(Display.VIRT, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, fps=100, to_ide=True)
elif DISPLAY_MODE == "LCD":
	Display.init(Display.ST7701, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, to_ide=True)
MediaManager.init()
sensor.run()
canny_thresh1	  = 50
canny_thresh2	  = 150
approx_epsilon	 = 0.04
area_min_ratio	 = 0.005
max_angle_cos	  = 0.3
gaussian_blur_size = 3
MIN_AREA = 500
MAX_AREA = 100000
MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 2.0
RECT_WIDTH = 210
RECT_HEIGHT = 95
BASE_RADIUS = 45
POINTS_PER_CIRCLE = 24
PURPLE_THRESHOLD = (20, 60, 15, 70, -70, -20)
CONFIRM_FRAMES = 1
LOST_FRAMES = 5
frame_seq = 0
if hasattr(time, "ticks_ms"):
	def get_obs_ms():
		return time.ticks_ms() & 0xFFFFFFFF
else:
	def get_obs_ms():
		return int(time.time() * 1000) & 0xFFFFFFFF
stable_count = 0
lost_count = 0
def calculate_distance(p1, p2):
	return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
def calculate_center(points):
	if not points: return (0, 0)
	return (sum(p[0] for p in points)/len(points), sum(p[1] for p in points)/len(points))
def is_valid_rect(corners):
	edges = [calculate_distance(corners[i], corners[(i+1)%4]) for i in range(4)]
	ratio1 = edges[0] / max(edges[2], 0.1)
	ratio2 = edges[1] / max(edges[3], 0.1)
	valid_ratio = 0.5 < ratio1 < 1.5 and 0.5 < ratio2 < 1.5
	area = abs(sum(corners[i][0]*corners[(i+1)%4][1] - corners[(i+1)%4][0]*corners[i][1] for i in range(4))) / 2
	valid_area = MIN_AREA < area < MAX_AREA
	xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
	w, h = max(xs)-min(xs), max(ys)-min(ys)
	valid_aspect = MIN_ASPECT_RATIO < w/max(h,0.1) < MAX_ASPECT_RATIO
	return valid_ratio and valid_area and valid_aspect
def sort_corners(corners):
	center = calculate_center(corners)
	sorted_pts = sorted(corners, key=lambda p: math.atan2(p[1]-center[1], p[0]-center[0]))
	if len(sorted_pts) == 4:
		left_top = min(sorted_pts, key=lambda p: p[0]+p[1])
		idx = sorted_pts.index(left_top)
		sorted_pts = sorted_pts[idx:] + sorted_pts[:idx]
	return sorted_pts
def get_perspective_matrix(src_pts, dst_pts):
	A = []; B = []
	for i in range(4):
		x,y = src_pts[i]; u,v = dst_pts[i]
		A.append([x,y,1,0,0,0,-u*x,-u*y])
		A.append([0,0,0,x,y,1,-v*x,-v*y])
		B.append(u); B.append(v)
	n = 8
	for i in range(n):
		pivot = i
		for j in range(i,n):
			if abs(A[j][i]) > abs(A[pivot][i]): pivot = j
		A[i], A[pivot] = A[pivot], A[i]
		B[i], B[pivot] = B[pivot], B[i]
		if abs(A[i][i]) < 1e-8: return None
		div = A[i][i]
		for j in range(i,n): A[i][j] /= div
		B[i] /= div
		for j in range(n):
			if j!=i and A[j][i]!=0:
				fac = A[j][i]
				for k in range(i,n): A[j][k] -= fac*A[i][k]
				B[j] -= fac*B[i]
	return [[B[0],B[1],B[2]], [B[3],B[4],B[5]], [B[6],B[7],1.0]]
def transform_points(points, matrix):
	res = []
	for x,y in points:
		xh = x*matrix[0][0] + y*matrix[0][1] + matrix[0][2]
		yh = x*matrix[1][0] + y*matrix[1][1] + matrix[1][2]
		wh = x*matrix[2][0] + y*matrix[2][1] + matrix[2][2]
		if abs(wh)>1e-8: res.append((xh/wh, yh/wh))
	return res
def detect_purple_blobs(img):
	return img.find_blobs([PURPLE_THRESHOLD], pixels_threshold=100, area_threshold=100, merge=True)
clock = time.clock()
image_shape = [sensor.height(), sensor.width()]
while True:
	clock.tick()
	img = sensor.snapshot()
	purple_blobs = detect_purple_blobs(img)
	for blob in purple_blobs:
		img.draw_rectangle(blob[0:4], color=(255,0,255), thickness=1)
		img.draw_cross(blob.cx(), blob.cy(), color=(255,0,255), thickness=1)
	gray_img = img.to_grayscale()
	img_np = gray_img.to_numpy_ref()
	rects = cv_lite.grayscale_find_rectangles_with_corners(
		image_shape, img_np,
		canny_thresh1, canny_thresh2,
		approx_epsilon, area_min_ratio,
		max_angle_cos, gaussian_blur_size
	)
	max_area = 0
	best_rect_corners = None
	for rect in rects:
		corners = [(rect[4], rect[5]), (rect[6], rect[7]),
				   (rect[8], rect[9]), (rect[10], rect[11])]
		if is_valid_rect(corners):
			area = rect[2] * rect[3]
			if area > max_area:
				max_area = area
				best_rect_corners = corners
	if best_rect_corners is not None:
		sorted_corners = sort_corners(best_rect_corners)
		raw_center = calculate_center(sorted_corners)
		stable_count += 1
		lost_count = 0
		if stable_count >= CONFIRM_FRAMES:
			for i in range(4):
				x1,y1 = sorted_corners[i]
				x2,y2 = sorted_corners[(i+1)%4]
				img.draw_line(x1,y1,x2,y2, color=(255,0,0), thickness=2)
			for p in sorted_corners:
				img.draw_circle(p[0], p[1], 5, color=(0,255,0), thickness=2)
			virtual_rect = [(0,0),(RECT_WIDTH,0),(RECT_WIDTH,RECT_HEIGHT),(0,RECT_HEIGHT)]
			matrix = get_perspective_matrix(virtual_rect, sorted_corners)
			if matrix:
				virtual_center = (RECT_WIDTH/2, RECT_HEIGHT/2)
				vir_pts = []
				for i in range(POINTS_PER_CIRCLE):
					ang = 2*math.pi*i/POINTS_PER_CIRCLE
					vir_pts.append((virtual_center[0]+BASE_RADIUS*math.cos(ang),
									virtual_center[1]+BASE_RADIUS*math.sin(ang)))
				mapped_pts = transform_points(vir_pts, matrix)
				for px, py in [(int(round(x)), int(round(y))) for x,y in mapped_pts]:
					img.draw_circle(px, py, 2, color=(255,0,255), thickness=2)
				mapped_center = transform_points([virtual_center], matrix)
				if mapped_center:
					mc = mapped_center[0]
					img.draw_circle(int(round(mc[0])), int(round(mc[1])), 3, color=(0,0,255), thickness=1)
			raw_x = int(round(raw_center[0]))
			raw_y = int(round(raw_center[1]))
			if 0 <= raw_x < sensor.width() and 0 <= raw_y < sensor.height():
				try:
					pto_data = pto.get_color_data(raw_x, raw_y, 0, 0)
					uart1.send(pto_data)
					obs_data = "@OBS,{},{:d}\r\n".format(frame_seq, get_obs_ms())
					uart1.send(obs_data)
					frame_seq = (frame_seq + 1) & 0xFFFF
				except Exception:
					pass
	else:
		stable_count = 0
		lost_count += 1
	cam_center_x = sensor.width() // 2
	cam_center_y = sensor.height() // 2
	img.draw_cross(cam_center_x, cam_center_y, color=(0, 255, 0), size=10, thickness=2)
	fps = clock.fps()
	img.draw_string_advanced(10, 10, 20, f"FPS: {fps:.1f}", color=(255, 255, 255))
	Display.show_image(img,
					  x=round((lcd_width - sensor.width()) / 2),
					  y=round((lcd_height - sensor.height()) / 2))
	print(f"FPS: {fps:.1f}")
	gc.collect()