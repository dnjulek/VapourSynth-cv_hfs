#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/hfs.hpp>

#include <VapourSynth4.h>
#include <VSHelper4.h>

struct HFSData final {
	VSNode* node;
	const VSVideoInfo* vi;
	float 	segEgbThresholdI;
	int 	minRegionSizeI;
	float 	segEgbThresholdII;
	int 	minRegionSizeII;
	float 	spatialWeight;
	int 	slicSpixelSize;
	int 	numSlicIter;
};

static void hfs_process_rgb(const VSFrame* src, VSFrame* dst, const HFSData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept {
	const int w = vsapi->getFrameWidth(src, 0);
	const int h = vsapi->getFrameHeight(src, 0);
	ptrdiff_t stride = vsapi->getStride(src, 0);

	const uint8_t* srcp_r = vsapi->getReadPtr(src, 0);
	const uint8_t* srcp_g = vsapi->getReadPtr(src, 1);
	const uint8_t* srcp_b = vsapi->getReadPtr(src, 2);
	uint8_t* dstp_r = vsapi->getWritePtr(dst, 0);
	uint8_t* dstp_g = vsapi->getWritePtr(dst, 1);
	uint8_t* dstp_b = vsapi->getWritePtr(dst, 2);

	cv::Mat srcImg_b(cv::Size(w, h), CV_8UC1);
	cv::Mat srcImg_g(cv::Size(w, h), CV_8UC1);
	cv::Mat srcImg_r(cv::Size(w, h), CV_8UC1);

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			srcImg_b.at<uint8_t>(y, x) = srcp_b[x];
		}
		srcp_b += stride;
	}

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			srcImg_g.at<uint8_t>(y, x) = srcp_g[x];
		}
		srcp_g += stride;
	}

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			srcImg_r.at<uint8_t>(y, x) = srcp_r[x];
		}
		srcp_r += stride;
	}


	cv::Mat srcImg_bgr(cv::Size(w, h), CV_8UC3);
	cv::Mat dstImg_bgr(cv::Size(w, h), CV_8UC3);
	std::vector<cv::Mat> src_channels(3);
	std::vector<cv::Mat> dst_channels(3);

	srcImg_b.convertTo(src_channels[0], CV_8U);
	srcImg_g.convertTo(src_channels[1], CV_8U);
	srcImg_r.convertTo(src_channels[2], CV_8U);

	merge(src_channels, srcImg_bgr);
	cv::Ptr<cv::hfs::HfsSegment> seg = cv::hfs::HfsSegment::create(h, w, d->segEgbThresholdI, d->minRegionSizeI, d->segEgbThresholdII, d->minRegionSizeII, d->spatialWeight, d->slicSpixelSize, d->numSlicIter);
	dstImg_bgr = seg->performSegmentCpu(srcImg_bgr);

	split(dstImg_bgr, dst_channels);
	cv::Mat dstImg_b = dst_channels[0];
	cv::Mat dstImg_g = dst_channels[1];
	cv::Mat dstImg_r = dst_channels[2];

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			dstp_b[x] = dstImg_b.at<uint8_t>(y, x);
		}
		dstp_b += stride;
	}

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			dstp_g[x] = dstImg_g.at<uint8_t>(y, x);
		}
		dstp_g += stride;
	}

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			dstp_r[x] = dstImg_r.at<uint8_t>(y, x);
		}
		dstp_r += stride;
	}
}

static const VSFrame* VS_CC hfsGetFrame(int n, int activationReason, void* instanceData, void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
	auto d{ static_cast<HFSData*>(instanceData) };

	if (activationReason == arInitial) {
		vsapi->requestFrameFilter(n, d->node, frameCtx);
	}
	else if (activationReason == arAllFramesReady) {
		const VSFrame* src = vsapi->getFrameFilter(n, d->node, frameCtx);

		const VSVideoFormat* fi = vsapi->getVideoFrameFormat(src);
		int height = vsapi->getFrameHeight(src, 0);
		int width = vsapi->getFrameWidth(src, 0);
		VSFrame* dst = vsapi->newVideoFrame(fi, width, height, src, core);

		hfs_process_rgb(src, dst, d, vsapi);

		vsapi->freeFrame(src);
		return dst;
	}
	return nullptr;
}

static void VS_CC hfsFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
	auto d{ static_cast<HFSData*>(instanceData) };
	vsapi->freeNode(d->node);
	delete d;
}

void VS_CC hfsCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
	auto d{ std::make_unique<HFSData>() };
	int err{ 0 };

	d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
	d->vi = vsapi->getVideoInfo(d->node);

	d->segEgbThresholdI = vsapi->mapGetFloatSaturated(in, "segEgbThresholdI", 0, &err);
	if (err)
		d->segEgbThresholdI = 0.08f;

	d->minRegionSizeI = vsapi->mapGetIntSaturated(in, "minRegionSizeI", 0, &err);
	if (err)
		d->minRegionSizeI = 100;

	d->segEgbThresholdII = vsapi->mapGetFloatSaturated(in, "segEgbThresholdII", 0, &err);
	if (err)
		d->segEgbThresholdII = 0.28f;

	d->minRegionSizeII = vsapi->mapGetIntSaturated(in, "minRegionSizeII", 0, &err);
	if (err)
		d->minRegionSizeII = 200;

	d->spatialWeight = vsapi->mapGetFloatSaturated(in, "spatialWeight", 0, &err);
	if (err)
		d->spatialWeight = 0.6f;

	d->slicSpixelSize = vsapi->mapGetIntSaturated(in, "slicSpixelSize", 0, &err);
	if (err)
		d->slicSpixelSize = 8;

	d->numSlicIter = vsapi->mapGetIntSaturated(in, "numSlicIter", 0, &err);
	if (err)
		d->numSlicIter = 5;


	if (d->vi->format.bytesPerSample != 1 || (d->vi->format.colorFamily != cfRGB)) {
		vsapi->mapSetError(out, "HFS: only RGB24 format is supported.");
		vsapi->freeNode(d->node);
		return;
	}

	VSFilterDependency deps[] = { {d->node, rpGeneral} };
	vsapi->createVideoFilter(out, "HFS", d->vi, hfsGetFrame, hfsFree, fmParallel, deps, 1, d.get(), core);
	d.release();
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
	vspapi->configPlugin("com.julek.cv_hfs", "cv_hfs", "Image Segmentation", VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
	vspapi->registerFunction("HFS",
		"clip:vnode;"
		"segEgbThresholdI:float:opt;"
		"minRegionSizeI:int:opt;"
		"segEgbThresholdII:float:opt;"
		"minRegionSizeII:int:opt;"
		"spatialWeight:float:opt;"
		"slicSpixelSize:int:opt;"
		"numSlicIter:int:opt;",
		"clip:vnode;",
		hfsCreate, nullptr, plugin);
}