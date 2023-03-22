from django.shortcuts import render, HttpResponse
from django.http import FileResponse
from server.algorithm import logzip
from django.views.decorators.csrf import csrf_exempt
from .models import User
from django.http import JsonResponse
import os
import re
import logging
import time


# Create your views here.
def introduce(request):
    return render(request, 'index.html')


def compress_page(request):
    return render(request, 'compress.html')


def decompress_page(request):
    return render(request, 'decompress.html')


def show_template(request):
    print(request.GET)
    filename = request.GET['filename']
    return render(request, 'tree.html', {'name': filename})


def get_file(request):
    log_data = request.FILES['log_file']
    target_dir = 'raw' if request.POST['type'] == 'log' else 'zipped'
    with open(log_path := os.path.join('server', 'logs', target_dir, log_data.name), 'wb') as log_file:
        for chunk in log_data.chunks():
            log_file.write(chunk)
    if request.POST['type'] == 'log':
        with open(log_path, 'r') as log_file:
            reg_demo = [log_file.readline()[:-1] for i in range(3)]
        return HttpResponse('<br>'.join(reg_demo))
    return HttpResponse('success')


def decompress_zip(request):
    start = time.time()
    logzip.decompress(os.path.join('server', 'logs', 'zipped', request.GET['filename']),
                      os.path.join('server', 'logs', 'decompressed'),' ')
    end = time.time()
    runTime = end - start
    print("解压缩运行时间：", runTime, "秒")
    return HttpResponse('success')


def download_dec(request):
    response = FileResponse(open(os.path.join('server', 'logs', 'decompressed', request.GET['filename']), 'rb'))
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = f"attachment;filename={request.GET['filename']}"
    return response


def download_zip(request):
    response = FileResponse(open(os.path.join('server', 'logs', 'zipped', request.GET['filename']), 'rb'))
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = f"attachment;filename={request.GET['filename']}"
    return response


def compress_log_file(request):
    start = time.time()
    # 返回分离消息头和消息框架的部分结果，将其展示在前端
    raw_log_path = os.path.join('server', 'logs', 'raw', request.POST['filename'])
    if not os.path.exists(raw_log_path):
        return HttpResponse('还未上传文件，无法压缩')
    with open(raw_log_path, 'r') as log_file:
        reg_demo = [log_file.readline()[:-1] for i in range(3)]
    try:
        pattern = re.compile(request.POST['reg'])
        lines = ['<span style="color: green">压缩成功!</span>']
        for log in reg_demo:
            result = pattern.search(log)
            head = '<span style="color: purple"> | </span>'.join(
                [f'<span style="color: red;text-decoration: underline">{piece}</span>' for piece in result.groups()])
            content = f'<span style="color: cornflowerblue">{log[result.span()[1]:]}</span>'
            lines.append(head + content)
    except Exception as e:
        return HttpResponse('正则表达式分离消息头时出错!<br>' + str(e))
    # 调用压缩算法
    log_file = open(os.path.join('server', 'logs', 'run.log'), encoding='utf-8', mode='a')
    logging.basicConfig(stream=log_file,
                        level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S %p")
    compressor = logzip.LogZip(raw_log_path,
                               os.path.join('server', 'logs', 'zipped'),
                               request.POST['delimiter']
                               )
    compressor.zip(float(request.POST['rate']), float(request.POST['threshold']), int(request.POST['top_n']),
                   pattern)
    filename = request.POST['filename']
    compressor.prefix_tree.create_chart(os.path.join('server', 'static', 'tree'), filename[:filename.rindex('.')])
    log_file.close()
    # 获取结束时间
    end = time.time()
    # 计算运行时间
    runTime = end - start
    # runTime_ms = runTime * 1000
    print("压缩运行时间：", runTime, "秒")
    # print("运行时间：", runTime_ms, "毫秒")
    return HttpResponse('<br>'.join(lines))


@csrf_exempt
def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = User.objects.filter(username=username, password=password).first()
        if user:
            return JsonResponse({'success': True})
        else:
            return JsonResponse({'success': False})
    return render(request, 'login.html')