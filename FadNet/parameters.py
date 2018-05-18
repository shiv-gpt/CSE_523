class trainParams():
	name = "default"
	img_sz = 64
	img_fm = 3
	attr = "Eyeglasses"
	instance_norm = False
	init_fm = 32
	max_fm = 512
	n_layers = 5
	n_skip = 0
	deconv_method = "convtranspose"
	hid_dim = 512
	dec_dropout = 0
	lat_dis_dropout = 0.3
	n_lat_dis = 1
	n_ptc_dis = 0
	n_clf_dis = 0
	smooth_label = 0.2
	lambda_ae = 1
	lambda_lat_dis = 0.0001
	lambda_ptc_dis = 0
	lambda_clf_dis = 0
	lambda_schedule = 500000
	v_flip = False
	h_flip = True
	batch_size = 50 #32
	ae_optimizer = "adam,lr=0.0002"
	dis_optimizer = "adam,lr=0.0002"
	clip_grad_norm = 5
	n_epochs = 30
	epoch_size = 50000 #50000
	ae_reload = ""
	lat_dis_reload = ""
	ptc_dis_reload = ""
	clf_dis_reload = ""
	eval_clf = ""
	debug = False


class interpolateParams():
	#model_path = "/home/shivang/FadNet/models/default/i6vh4zhnm3/final_ae.pth"
	#model_path = "/home/shivang/FadNet/models/default/keja5552y0/final_ae.pth"
	model_path = "/home/shivang/FadNet/models/default/r9jmkwt56d/final_ae.pth"
	#model_path = "/home/shivang/FadNet/models/default/d8fl8gszdw/final_ae.pth"
	#model_path = "/home/shivang/FadNet/models/default/elszu3xh2g/final_ae.pth"
	#model_path = "/home/shivang/FadNet/models/default/kxhdf7ylf4/final_ae.pth"
	#model_path = "/home/shivang/FadNet/models/default/7p9l24s0r2/final_ae.pth"
	#model_path = "/home/shivang/FadNet/models/default/4pdyuudmd3/final_ae.pth"
	#model_path = "/home/shivang/FadNet/models/default/zwhifssajb/final_ae.pth"
	#model_path = "/home/shivang/FadNet/models/default/v1izp79zc3/final_ae.pth"
	#model_path = "/home/shivang/FadNet/models/default/bm2sj6168e/final_ae.pth"
	#model_path = "/home/shivang/FadNet/models/default/exvchkuelg/final_ae.pth"
	n_images = 5 
	offset = 0
	n_interpolations = 5
	alpha_min = 2.0
	alpha_max = 2.0
	plot_size = 5
	row_wise = True
	output_path = "e_new"


