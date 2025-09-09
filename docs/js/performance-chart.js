document.addEventListener('DOMContentLoaded', function() {
    const ctx = document.getElementById('performanceChart').getContext('2d');
    
    const performanceData = {
        labels: [
            'Naive',
            'Global Memory Coalescing',
            'Shared Memory Blocking',
            '1D Block Tiling',
            '2D Block Tiling',
            'Vectorized Memory Access',
            'Bank Conflict Resolution',
            'Extra Column Padding',
            'Autotuned',
            'Warp Tiling',
            'Double Buffering',
            'TMA'
        ],
        datasets: [{
            label: 'Performance (GFLOP/s)',
            data: [50, 120, 340, 520, 680, 820, 950, 1050, 1180, 1280, 1350, 1580],
            backgroundColor: 'rgba(75, 192, 192, 0.7)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1
        }]
    };
    
    new Chart(ctx, {
        type: 'bar',
        data: performanceData,
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Convolution Performance Across Optimization Techniques (4096×4096 Matrix)',
                    font: {
                        size: 16
                    }
                },
                legend: {
                    position: 'top',
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'GFLOP/s'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Implementation'
                    }
                }
            }
        }
    });
});
