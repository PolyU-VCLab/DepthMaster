// Demo Examples Comparison (Perspective)
// 使用 Three.js 点云渲染器加载 PLY 文件，支持分页浏览和双方法对比
// 左侧固定 Ours，右侧选择对比方法
// 懒加载：滚动到可见区域时才初始化

let demoComparisonInitialized = false;

function initDemoComparison() {
    if (demoComparisonInitialized) return;
    demoComparisonInitialized = true;
    console.log('[懒加载] 初始化: Perspective Comparison');

    // 创建两个点云查看器
    const viewer1 = new PointCloudViewer('perspComparisonCanvas1', 'perspComparisonLoading1');
    const viewer2 = new PointCloudViewer('perspComparisonCanvas2', 'perspComparisonLoading2');
    allPanoViewers.push(viewer1, viewer2); // 加入全局列表以便 resize

    let allExamples = [];
    let currentPage = 1;
    const itemsPerPage = 8;
    let totalPages = 1;

    fetch('static/perspective_comparison.json')
        .then(response => response.json())
        .then(data => {
            allExamples = data;
            totalPages = Math.ceil(allExamples.length / itemsPerPage);
            renderPage(1);
            if (allExamples.length > 0) {
                loadComparison(allExamples[0]);
            }
            if (totalPages <= 1) {
                const paginationDiv = document.querySelector('#comparison_container .custom-carousel-container');
                if (paginationDiv) paginationDiv.style.display = 'none';
            }
        });

    function renderPage(page) {
        currentPage = page;
        const panel = document.getElementById('comparisonSelectionPanel');
        panel.innerHTML = '';

        const start = (page - 1) * itemsPerPage;
        const end = Math.min(start + itemsPerPage, allExamples.length);

        for (let i = start; i < end; i++) {
            const item = allExamples[i];
            const wrapper = document.createElement('div');
            wrapper.className = 'selectable-item';
            wrapper.setAttribute('data-index', i);

            const img = document.createElement('img');
            img.className = 'selectable-image';
            if (i === 0 && page === 1) img.classList.add('selected');
            img.setAttribute('data-index', i);
            img.setAttribute('name', item.name);
            img.src = `${PERSP_DATA_BASE}/${item.name}/${item.image}`;
            img.title = item.name;

            wrapper.appendChild(img);
            panel.appendChild(wrapper);
        }

        document.getElementById('pageCounter').innerText = `Page ${page} / ${totalPages}`;
    }

    // 左侧固定加载 ours，右侧加载选中的对比方法
    function loadComparison(item) {
        const method2 = document.getElementById('comparisonMethod2Selection').value;
        const basePath = `${PERSP_DATA_BASE}/${item.name}`;

        viewer1.resetView();
        viewer1.loadPLY(`${basePath}/ours/pointcloud.ply`, 'ours', false);

        viewer2.resetView();
        viewer2.loadPLY(`${basePath}/${method2}/pointcloud.ply`, method2, false);
    }

    function getSelectedExample() {
        const selectedImg = document.querySelector('#comparisonSelectionPanel .selectable-image.selected');
        if (!selectedImg) return allExamples[0];
        const idx = parseInt(selectedImg.getAttribute('data-index'));
        return allExamples[idx];
    }

    // 图片点击选择
    const comparisonSelectionPanel = document.getElementById('comparisonSelectionPanel');
    comparisonSelectionPanel.addEventListener('click', function(event) {
        const wrapper = event.target.closest('.selectable-item');
        if (!wrapper) return;
        const img = wrapper.querySelector('.selectable-image');
        if (!img || img.classList.contains('selected')) return;

        comparisonSelectionPanel.querySelectorAll('.selectable-image').forEach(function(image) {
            image.classList.remove('selected');
        });
        img.classList.add('selected');

        const idx = parseInt(img.getAttribute('data-index'));
        loadComparison(allExamples[idx]);
    });

    // 分页按钮
    document.getElementById('prevPage').addEventListener('click', () => {
        if (currentPage > 1) renderPage(currentPage - 1);
    });
    document.getElementById('nextPage').addEventListener('click', () => {
        if (currentPage < totalPages) renderPage(currentPage + 1);
    });

    // 右侧方法选择变化时重新加载
    document.getElementById('comparisonMethod2Selection').addEventListener('change', function() {
        const item = getSelectedExample();
        if (item) {
            const method2 = this.value;
            const basePath = `${PERSP_DATA_BASE}/${item.name}`;
            viewer2.resetView();
            viewer2.loadPLY(`${basePath}/${method2}/pointcloud.ply`, method2, false);
        }
    });

    // Geometry/Textured 切换按钮
    const toggleComparisonLeftButton = document.querySelector('#toggleTexturedComparison .toggle-left');
    const toggleComparisonRightButton = document.querySelector('#toggleTexturedComparison .toggle-right');

    toggleComparisonLeftButton.addEventListener('click', function() {
        toggleComparisonLeftButton.classList.add('active');
        toggleComparisonRightButton.classList.remove('active');
        viewer1.setTextured(false);
        viewer2.setTextured(false);
    });

    toggleComparisonRightButton.addEventListener('click', function() {
        toggleComparisonLeftButton.classList.remove('active');
        toggleComparisonRightButton.classList.add('active');
        viewer1.setTextured(true);
        viewer2.setTextured(true);
    });

    // 同步两个查看器的视角
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

// 懒加载：观察 comparison_container
(function() {
    const container = document.getElementById('comparison_container');
    if (!container) return;
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                initDemoComparison();
                observer.unobserve(entry.target);
            }
        });
    }, { rootMargin: '300px 0px' });
    observer.observe(container);
})();
