// DepthMaster 项目部署配置
// 大文件（3D模型、输入图片）托管在腾讯云 COS 上
// 小文件（HTML/CSS/JS/JSON）托管在 GitHub Pages 上

// 腾讯云 COS 基础 URL（末尾不带斜杠）
const COS_BASE_URL = 'https://depthmaster-data-1425687910.cos.ap-guangzhou.myqcloud.com';

// 透视图数据路径前缀（COS 上的目录结构与本地一致）
const PERSP_DATA_BASE = `${COS_BASE_URL}/static/demo_examples`;

// 全景图数据路径前缀
const PANO_DATA_BASE = `${COS_BASE_URL}/static/demo_examples_pano`;
