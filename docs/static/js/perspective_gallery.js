// Perspective Gallery
// 使用 Three.js 点云渲染器加载 PLY 文件，支持分页浏览，只展示 ours 方法
// 懒加载：滚动到可见区域时才初始化

let perspGalleryInitialized = false;

function initPerspGallery() {
    if (perspGalleryInitialized) return;
    perspGalleryInitialized = true;
    console.log('[懒加载] 初始化: Perspective Gallery');

    // 创建点云查看器
    const viewer = new PointCloudViewer('perspGalleryCanvas', 'perspGalleryLoading');
    allPanoViewers.push(viewer); // 加入全局列表以便 resize

    let perspGalleryExamples = [];
    let perspGalleryCurrentPage = 1;
    const perspGalleryItemsPerPage = 8;
    let perspGalleryTotalPages = 1;

    fetch('static/perspective_gallery.json')
        .then(response => response.json())
        .then(data => {
            perspGalleryExamples = data;
            perspGalleryTotalPages = Math.ceil(perspGalleryExamples.length / perspGalleryItemsPerPage);
            renderPerspGalleryPage(1);
            if (perspGalleryExamples.length > 0) {
                loadPerspGalleryModel(perspGalleryExamples[0]);
            }
            updatePerspGalleryPagination();
        });

    function renderPerspGalleryPage(page) {
        perspGalleryCurrentPage = page;
        const panel = document.getElementById('perspGallerySelectionPanel');
        panel.innerHTML = '';

        const start = (page - 1) * perspGalleryItemsPerPage;
        const end = Math.min(start + perspGalleryItemsPerPage, perspGalleryExamples.length);

        for (let i = start; i < end; i++) {
            const item = perspGalleryExamples[i];

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

        document.getElementById('perspGalleryPageCounter').innerText = `Page ${page} / ${perspGalleryTotalPages}`;
    }

    function loadPerspGalleryModel(item) {
        const basePath = `${PERSP_DATA_BASE}/${item.name}`;
        const plyPath = `${basePath}/ours/pointcloud.ply`;
        viewer.resetView();
        viewer.loadPLY(plyPath, 'ours', false);
    }

    function updatePerspGalleryPagination() {
        const paginationDiv = document.getElementById('perspGalleryPagination');
        if (perspGalleryTotalPages <= 1) {
            paginationDiv.style.display = 'none';
        } else {
            paginationDiv.style.display = '';
        }
    }

    // 图片点击选择
    const perspGalleryPanel = document.getElementById('perspGallerySelectionPanel');
    perspGalleryPanel.addEventListener('click', function(event) {
        const wrapper = event.target.closest('.selectable-item');
        if (!wrapper) return;
        const img = wrapper.querySelector('.selectable-image');
        if (!img || img.classList.contains('selected')) return;

        perspGalleryPanel.querySelectorAll('.selectable-image').forEach(function(image) {
            image.classList.remove('selected');
        });
        img.classList.add('selected');

        const idx = parseInt(img.getAttribute('data-index'));
        loadPerspGalleryModel(perspGalleryExamples[idx]);
    });

    // 分页按钮
    document.getElementById('perspGalleryPrevPage').addEventListener('click', () => {
        if (perspGalleryCurrentPage > 1) {
            renderPerspGalleryPage(perspGalleryCurrentPage - 1);
        }
    });

    document.getElementById('perspGalleryNextPage').addEventListener('click', () => {
        if (perspGalleryCurrentPage < perspGalleryTotalPages) {
            renderPerspGalleryPage(perspGalleryCurrentPage + 1);
        }
    });

    // Geometry/Textured 切换
    const togglePerspGalleryLeft = document.querySelector('#toggleTexturedPerspGallery .toggle-left');
    const togglePerspGalleryRight = document.querySelector('#toggleTexturedPerspGallery .toggle-right');

    togglePerspGalleryLeft.addEventListener('click', function() {
        togglePerspGalleryLeft.classList.add('active');
        togglePerspGalleryRight.classList.remove('active');
        viewer.setTextured(false);
    });

    togglePerspGalleryRight.addEventListener('click', function() {
        togglePerspGalleryLeft.classList.remove('active');
        togglePerspGalleryRight.classList.add('active');
        viewer.setTextured(true);
    });
}

// 懒加载：观察 persp_gallery_container
(function() {
    const container = document.getElementById('persp_gallery_container');
    if (!container) return;
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                initPerspGallery();
                observer.unobserve(entry.target);
            }
        });
    }, { rootMargin: '300px 0px' });
    observer.observe(container);
})();
