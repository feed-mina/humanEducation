<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Electricity Prediction</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }

        h1 {
            text-align: center;
            margin-bottom: 30px;
        }

        canvas {
            display: block;
            margin: 0 auto;
            width: 800px !important; /* 차트 너비를 800px로 조정 */
            height: 300px !important; /* 차트 높이 조정 */
        }

        table {
            margin: 20px auto;
            border-collapse: collapse;
            width: 80%;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }

        th, td {
            padding: 8px;
            text-align: center;
            border: 1px solid #ddd;
        }

        th {
            background-color: #f4f4f4;
            font-weight: bold;
        }

        tr:nth-child(even) {
            background-color: #f9f9f9;
        }

        tr:hover {
            background-color: #f1f1f1;
        }

        /* 테이블 폰트 크기 줄이기 */
        table {
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>7-Day Electricity Prediction</h1>
    
    <canvas id="predictionChart"></canvas>
    
    <table>
        <thead>
            <tr>
                <th>Date</th>
                <th>Predicted Wattage</th>
            </tr>
        </thead>
        <tbody>
            <c:forEach var="prediction" items="${predictions}">
                <tr>
                    <td>${prediction.date}</td>
                    <td>${prediction.predictedWattage}</td>
                </tr>
            </c:forEach>
        </tbody>
    </table>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            var ctx = document.getElementById('predictionChart').getContext('2d');
            var dates = [];
            var wattages = [];

            // predictions 데이터가 있을 때만 차트 데이터를 생성
            <c:if test="${not empty predictions}">
                <c:forEach var="prediction" items="${predictions}">
                    dates.push('${prediction.date}');
                    wattages.push(${prediction.predictedWattage}); 
                </c:forEach>
            </c:if>

            // 차트 생성
            var chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: dates,
                    datasets: [{
                        label: 'Predicted Wattage',
                        data: wattages,
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false, // 비율 조정 허용
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                font: {
                                    size: 16 // 차트의 폰트 크기를 키움
                                }
                            }
                        },
                        tooltip: {
                            bodyFont: {
                                size: 14 // 툴팁 폰트 크기 조정
                            }
                        }
                    }
                }
            });
        });
    </script>
</body>
</html>
