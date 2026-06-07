// 全景图三栏可视化：室外对比、室内对比、Gallery（仅展示 ours）
// 使用 Three.js 渲染点云，支持分页浏览
// 懒加载：各模块在滚动到可见区域时才初始化

const PANO_ITEMS_PER_PAGE = 8;

// ========== Three.js 点云渲染器封装 ==========
class PointCloudViewer {
    constructor(canvasId, loadingId) {
        this.canvas = document.getElementById(canvasId);
        this.loadingEl = document.getElementById(loadingId);
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);

        this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
        this.camera.position.set(0, 0, 3);

        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
        this.renderer.setPixelRatio(window.devicePixelRatio);

        this.controls = new THREE.OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.1;
        this.controls.rotateSpeed = 0.8;
        this.controls.zoomSpeed = 1.0;
        this.controls.panSpeed = 0.8;

        this.pointCloud = null;
        this.originalColors = null;
        this.animating = false;
        this.isTextured = true;
        this._loadId = 0;  // 用于取消过期的异步加载

        this._resize();
        this._animate();
    }

    _resize() {
        const parent = this.canvas.parentElement;
        const w = parent.clientWidth;
        const h = parent.clientHeight;
        this.canvas.width = w * window.devicePixelRatio;
        this.canvas.height = h * window.devicePixelRatio;
        this.canvas.style.width = w + 'px';
        this.canvas.style.height = h + 'px';
        this.renderer.setSize(w, h);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
    }

    _animate() {
        if (!this.animating) this.animating = true;
        requestAnimationFrame(() => this._animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    showLoading() { if (this.loadingEl) this.loadingEl.style.display = 'block'; }
    hideLoading() { if (this.loadingEl) this.loadingEl.style.display = 'none'; }

    loadPLY(url, method, isPano) {
        // isPano: true 为全景图坐标变换，false/undefined 为透视图坐标变换
        if (isPano === undefined) isPano = true;
        this.showLoading();
        // 递增 loadId，使之前未完成的加载回调失效
        const currentLoadId = ++this._loadId;
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
            this.pointCloud = null;
            this.originalColors = null;
        }

        const isOurs = (method === 'ours' || method === 'ours_v3');
        const loader = new THREE.PLYLoader();
        loader.load(url, (geometry) => {
            // 如果已经有更新的加载请求，丢弃本次结果
            if (currentLoadId !== this._loadId) {
                geometry.dispose();
                return;
            }
            // 再次清理场景中可能残留的旧点云（防止竞态）
            if (this.pointCloud) {
                this.scene.remove(this.pointCloud);
                this.pointCloud.geometry.dispose();
                this.pointCloud.material.dispose();
                this.pointCloud = null;
                this.originalColors = null;
            }
            const pos = geometry.attributes.position;
            if (isPano) {
                // 全景图坐标变换
                for (let i = 0; i < pos.count; i++) {
                    let x = pos.getX(i);
                    let y = pos.getY(i);
                    let z = pos.getZ(i);
                    let tx = x, ty = -y, tz = -z;
                    if (!isOurs) {
                        const rx = -tz;
                        const rz = tx;
                        tx = rx;
                        tz = rz;
                    }
                    pos.setX(i, tx);
                    pos.setY(i, ty);
                    pos.setZ(i, tz);
                }
            } else {
                // 透视图：PLY 已在推理脚本中完成 OpenCV→OpenGL 坐标变换 [1,-1,-1]
                // Three.js 本身使用 OpenGL 坐标系，无需额外变换
            }
            pos.needsUpdate = true;

            geometry.computeBoundingBox();
            const box = geometry.boundingBox;
            const center = new THREE.Vector3();
            box.getCenter(center);
            geometry.translate(-center.x, -center.y, -center.z);

            const size = new THREE.Vector3();
            box.getSize(size);
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 2.0 / maxDim;
            geometry.scale(scale, scale, scale);

            if (geometry.attributes.color) {
                this.originalColors = geometry.attributes.color.array.slice();
            }

            const material = new THREE.PointsMaterial({
                size: 0.003,
                vertexColors: true,
                sizeAttenuation: true,
            });

            this.pointCloud = new THREE.Points(geometry, material);
            this.scene.add(this.pointCloud);
            this.setTextured(this.isTextured);
            this.hideLoading();
        }, undefined, (error) => {
            if (currentLoadId !== this._loadId) return;
            console.error('Error loading PLY:', error);
            this.hideLoading();
        });
    }

    setTextured(isTextured) {
        this.isTextured = isTextured;
        if (!this.pointCloud || !this.pointCloud.geometry.attributes.color) return;
        const colors = this.pointCloud.geometry.attributes.color;
        if (isTextured) {
            if (this.originalColors) {
                colors.array.set(this.originalColors);
                colors.needsUpdate = true;
            }
        } else {
            const arr = colors.array;
            for (let i = 0; i < arr.length; i += 3) {
                arr[i] = 0.6; arr[i + 1] = 0.6; arr[i + 2] = 0.6;
            }
            colors.needsUpdate = true;
        }
    }

    resetView() {
        this.camera.position.set(0, 0, 3);
        this.camera.lookAt(0, 0, 0);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    // 全景图 cubemap 正视角：相机从 -z 方向看向原点（对应 Front face）
    // 原始坐标系中 Front face 朝 +z，经过 ours 变换 (tx=x, ty=-y, tz=-z) 后变为 -z
    resetViewFront() {
        this.camera.position.set(0, 0, -3);
        this.camera.lookAt(0, 0, 0);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    getCameraState() {
        return {
            position: this.camera.position.clone(),
            target: this.controls.target.clone(),
            zoom: this.camera.zoom
        };
    }

    setCameraState(state) {
        this.camera.position.copy(state.position);
        this.controls.target.copy(state.target);
        this.camera.zoom = state.zoom;
        this.camera.updateProjectionMatrix();
        this.controls.update();
    }

    dispose() {
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
        }
        this.renderer.dispose();
        this.controls.dispose();
    }
}

// ========== 通用：创建对比面板逻辑（左侧固定 ours，右侧选择对比方法） ==========
function createComparisonPanel(config) {
    const {
        jsonUrl,
        panelId,
        method2SelectId,
        toggleContainerId,
        viewer1,
        viewer2,
        useFrontView  // 是否使用 cubemap 正视角作为初始视角
    } = config;

    let examples = [];
    let currentItem = null;
    let isTextured = true;

    // 渲染所有图片
    function renderAll() {
        const panel = document.getElementById(panelId);
        panel.innerHTML = '';

        for (let i = 0; i < examples.length; i++) {
            const item = examples[i];
            const wrapper = document.createElement('div');
            wrapper.className = 'selectable-item';
            wrapper.setAttribute('data-index', i);

            const img = document.createElement('img');
            img.className = 'selectable-image';
            if (currentItem && currentItem.name === item.name) img.classList.add('selected');
            else if (!currentItem && i === 0) img.classList.add('selected');
            img.setAttribute('data-index', i);
            img.setAttribute('name', item.name);
            img.src = `${PANO_DATA_BASE}/${item.name}/${item.image}`;
            img.title = item.name;

            wrapper.appendChild(img);
            panel.appendChild(wrapper);
        }
    }

    // 加载对比点云：左侧固定 ours，右侧加载选中的方法
    function loadComparison(item) {
        currentItem = item;
        const method2 = document.getElementById(method2SelectId).value;
        const basePath = `${PANO_DATA_BASE}/${item.name}`;

        if (viewer1) {
            if (useFrontView) viewer1.resetViewFront(); else viewer1.resetView();
            viewer1.loadPLY(`${basePath}/ours/pointcloud.ply`, 'ours');
        }
        if (viewer2) {
            if (useFrontView) viewer2.resetViewFront(); else viewer2.resetView();
            viewer2.loadPLY(`${basePath}/${method2}/pointcloud.ply`, method2);
        }
    }

    function getSelectedExample() {
        const selectedImg = document.querySelector(`#${panelId} .selectable-image.selected`);
        if (!selectedImg) return examples[0];
        const idx = parseInt(selectedImg.getAttribute('data-index'));
        return examples[idx];
    }

    // 加载数据
    fetch(jsonUrl)
        .then(r => r.json())
        .then(data => {
            examples = data;
            renderAll();
            if (examples.length > 0) {
                currentItem = examples[0];
                loadComparison(currentItem);
            }
        });

    // 图片点击
    document.getElementById(panelId).addEventListener('click', function(event) {
        const wrapper = event.target.closest('.selectable-item');
        if (!wrapper) return;
        const img = wrapper.querySelector('.selectable-image');
        if (!img || img.classList.contains('selected')) return;
        this.querySelectorAll('.selectable-image').forEach(i => i.classList.remove('selected'));
        img.classList.add('selected');
        const idx = parseInt(img.getAttribute('data-index'));
        loadComparison(examples[idx]);
    });

    // 右侧方法选择变化
    document.getElementById(method2SelectId).addEventListener('change', function() {
        const item = getSelectedExample();
        if (item && viewer2) {
            if (useFrontView) viewer2.resetViewFront(); else viewer2.resetView();
            viewer2.loadPLY(`${PANO_DATA_BASE}/${item.name}/${this.value}/pointcloud.ply`, this.value);
        }
    });

    // Textured 切换
    const toggleLeft = document.querySelector(`#${toggleContainerId} .toggle-left`);
    const toggleRight = document.querySelector(`#${toggleContainerId} .toggle-right`);
    toggleLeft.addEventListener('click', function() {
        toggleLeft.classList.add('active');
        toggleRight.classList.remove('active');
        isTextured = false;
        if (viewer1) viewer1.setTextured(false);
        if (viewer2) viewer2.setTextured(false);
    });
    toggleRight.addEventListener('click', function() {
        toggleLeft.classList.remove('active');
        toggleRight.classList.add('active');
        isTextured = true;
        if (viewer1) viewer1.setTextured(true);
        if (viewer2) viewer2.setTextured(true);
    });

    // 视角同步
    if (viewer1 && viewer2) {
        let syncSource = null;
        function syncView() {
            if (!syncSource) return;
            const source = syncSource === 1 ? viewer1 : viewer2;
            const target = syncSource === 1 ? viewer2 : viewer1;
            target.setCameraState(source.getCameraState());
        }
        viewer1.controls.addEventListener('change', () => { if (syncSource === 1) syncView(); });
        viewer2.controls.addEventListener('change', () => { if (syncSource === 2) syncView(); });
        viewer1.canvas.addEventListener('mousedown', () => { syncSource = 1; });
        viewer1.canvas.addEventListener('wheel', () => { syncSource = 1; });
        viewer2.canvas.addEventListener('mousedown', () => { syncSource = 2; });
        viewer2.canvas.addEventListener('wheel', () => { syncSource = 2; });
    }
}

// ========== 通用：创建 Gallery 面板逻辑（单个查看器，仅 ours） ==========
function createGalleryPanel(config) {
    const {
        jsonUrl,
        panelId,
        pageCounterId,
        prevPageId,
        nextPageId,
        paginationId,
        toggleContainerId,
        viewer
    } = config;

    let examples = [];
    let currentPage = 1;
    let totalPages = 1;
    let currentItem = null;
    let isTextured = true;

    function renderPage(page) {
        currentPage = page;
        const panel = document.getElementById(panelId);
        panel.innerHTML = '';
        const start = (page - 1) * PANO_ITEMS_PER_PAGE;
        const end = Math.min(start + PANO_ITEMS_PER_PAGE, examples.length);

        for (let i = start; i < end; i++) {
            const item = examples[i];
            const wrapper = document.createElement('div');
            wrapper.className = 'selectable-item';
            wrapper.setAttribute('data-index', i);

            const img = document.createElement('img');
            img.className = 'selectable-image';
            if (currentItem && currentItem.name === item.name) img.classList.add('selected');
            else if (!currentItem && i === 0 && page === 1) img.classList.add('selected');
            img.setAttribute('data-index', i);
            img.setAttribute('name', item.name);
            img.src = `${PANO_DATA_BASE}/${item.name}/${item.image}`;
            img.title = item.name;

            wrapper.appendChild(img);
            panel.appendChild(wrapper);
        }
        document.getElementById(pageCounterId).innerText = `Page ${page} / ${totalPages}`;
    }

    function loadScene(item) {
        currentItem = item;
        const basePath = `${PANO_DATA_BASE}/${item.name}`;
        if (viewer) {
            viewer.resetView();
            viewer.loadPLY(`${basePath}/ours/pointcloud.ply`, 'ours');
        }
    }

    fetch(jsonUrl)
        .then(r => r.json())
        .then(data => {
            examples = data;
            totalPages = Math.ceil(examples.length / PANO_ITEMS_PER_PAGE);
            renderPage(1);
            if (examples.length > 0) {
                currentItem = examples[0];
                loadScene(currentItem);
            }
            // 只有一页时隐藏分页控件
            const paginationEl = document.getElementById(paginationId);
            if (paginationEl && totalPages <= 1) {
                paginationEl.style.display = 'none';
            }
        });

    document.getElementById(panelId).addEventListener('click', function(event) {
        const wrapper = event.target.closest('.selectable-item');
        if (!wrapper) return;
        const img = wrapper.querySelector('.selectable-image');
        if (!img || img.classList.contains('selected')) return;
        this.querySelectorAll('.selectable-image').forEach(i => i.classList.remove('selected'));
        img.classList.add('selected');
        const idx = parseInt(img.getAttribute('data-index'));
        loadScene(examples[idx]);
    });

    document.getElementById(prevPageId).addEventListener('click', () => {
        if (currentPage > 1) renderPage(currentPage - 1);
    });
    document.getElementById(nextPageId).addEventListener('click', () => {
        if (currentPage < totalPages) renderPage(currentPage + 1);
    });

    const toggleLeft = document.querySelector(`#${toggleContainerId} .toggle-left`);
    const toggleRight = document.querySelector(`#${toggleContainerId} .toggle-right`);
    toggleLeft.addEventListener('click', function() {
        toggleLeft.classList.add('active');
        toggleRight.classList.remove('active');
        isTextured = false;
        if (viewer) viewer.setTextured(false);
    });
    toggleRight.addEventListener('click', function() {
        toggleLeft.classList.remove('active');
        toggleRight.classList.add('active');
        isTextured = true;
        if (viewer) viewer.setTextured(true);
    });
}

// ========== 懒加载：使用 IntersectionObserver 延迟初始化各面板 ==========
// 记录所有创建的 viewer 以便 resize
const allPanoViewers = [];

function initPanoGallery() {
    const galleryViewer = new PointCloudViewer('panoGalleryCanvas', 'panoGalleryLoading');
    allPanoViewers.push(galleryViewer);
    createGalleryPanel({
        jsonUrl: 'static/pano_gallery.json',
        panelId: 'panoGallerySelectionPanel',
        pageCounterId: 'panoGalleryPageCounter',
        prevPageId: 'panoGalleryPrevPage',
        nextPageId: 'panoGalleryNextPage',
        paginationId: 'panoGalleryPagination',
        toggleContainerId: 'toggleTexturedPanoGallery',
        viewer: galleryViewer
    });
}

function initPanoIndoor() {
    const indoorViewer1 = new PointCloudViewer('panoIndoorCanvas1', 'panoIndoorLoading1');
    const indoorViewer2 = new PointCloudViewer('panoIndoorCanvas2', 'panoIndoorLoading2');
    allPanoViewers.push(indoorViewer1, indoorViewer2);
    createComparisonPanel({
        jsonUrl: 'static/pano_indoor.json',
        panelId: 'panoIndoorSelectionPanel',
        method2SelectId: 'panoIndoorMethod2Selection',
        toggleContainerId: 'toggleTexturedPanoIndoor',
        viewer1: indoorViewer1,
        viewer2: indoorViewer2
    });
}

function initPanoOutdoor() {
    const outdoorViewer1 = new PointCloudViewer('panoOutdoorCanvas1', 'panoOutdoorLoading1');
    const outdoorViewer2 = new PointCloudViewer('panoOutdoorCanvas2', 'panoOutdoorLoading2');
    allPanoViewers.push(outdoorViewer1, outdoorViewer2);
    createComparisonPanel({
        jsonUrl: 'static/pano_outdoor.json',
        panelId: 'panoOutdoorSelectionPanel',
        method2SelectId: 'panoOutdoorMethod2Selection',
        toggleContainerId: 'toggleTexturedPanoOutdoor',
        viewer1: outdoorViewer1,
        viewer2: outdoorViewer2,
        useFrontView: true  // outdoor 使用 cubemap 正视角
    });
}

// 懒加载入口：观察各容器，滚动到附近时才初始化
function setupLazyLoading() {
    const lazyTargets = [
        { elementId: 'pano_gallery_container', initFn: initPanoGallery },
        { elementId: 'pano_indoor_container', initFn: initPanoIndoor },
        { elementId: 'pano_outdoor_container', initFn: initPanoOutdoor },
    ];

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = lazyTargets.find(t => t.element === entry.target);
                if (target && !target.initialized) {
                    target.initialized = true;
                    console.log(`[懒加载] 初始化: ${target.elementId}`);
                    target.initFn();
                    observer.unobserve(entry.target);
                }
            }
        });
    }, {
        rootMargin: '300px 0px'  // 提前 300px 开始加载
    });

    lazyTargets.forEach(target => {
        target.element = document.getElementById(target.elementId);
        target.initialized = false;
        if (target.element) {
            observer.observe(target.element);
        }
    });

    // 窗口大小变化时 resize 所有已创建的 viewer
    window.addEventListener('resize', () => {
        allPanoViewers.forEach(v => v._resize());
    });
}

// 启动懒加载
setupLazyLoading();
