from numpy import select
from other_gnn_models import LINKX, H2GCN, APPNP_Net
from glognn import MLP_NORM

def parse_method(args, n, c, d, device, edge_index=None):
    if args.method == 'linkx':
        model = LINKX(
            in_channels=d, hidden_channels=args.hidden_channels, 
            out_channels=c, num_layers=args.num_layers, num_nodes=n,
            inner_activation=args.inner_activation, 
            inner_dropout=args.inner_dropout, 
            dropout=args.dropout, 
            init_layers_A=args.link_init_layers_A, 
            init_layers_X=args.link_init_layers_X).to(device)
    elif args.method == 'appnp':
        model = APPNP_Net(
            in_channels=d, hidden_channels=args.hidden_channels, out_channels=c, 
            alpha=args.gpr_alpha, dropout=args.dropout, num_layers=args.num_layers).to(device)
    elif args.method == 'h2gcn':
        if edge_index == None:
            raise ValueError('Should provide edge index')
        model = H2GCN(
            in_channels=d, hidden_channels=args.hidden_channels, 
            out_channels=c, edge_index=edge_index, num_nodes=n,
            num_layers=args.num_layers, dropout=args.dropout,
            num_mlp_layers=args.num_mlp_layers).to(device)
    elif args.method == 'mlpnorm':
        model = MLP_NORM(
            nnodes=n, nfeat=d, nhid=args.hidden_channels, 
            nclass=c, dropout=args.dropout, alpha=args.alpha, beta=args.beta, gamma=args.gamma,
            delta=args.delta, norm_func_id=args.norm_func_id, norm_layers=args.norm_layers, 
            orders_func_id=args.orders_func_id, orders=args.orders, cuda=True).to(device)
    else:
        raise ValueError('Invalid method')
    return model

def parser_add_main_args(parser):
    parser.add_argument('--dataset', type=str, default='tolokers', help='dataset name')
    # model training parameters
    parser.add_argument('--cuda', type=int, default=0, help='Avaiable GPU ID')
    parser.add_argument('--method', type=str, default='acm', help='which model to use')
    parser.add_argument('--run', type=int, default=10, help='number of graph per homophily level')
    parser.add_argument('--epoch_num', type=int, default=1000, help='Number of Epoch')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden dim')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    #
    parser.add_argument('--early_stopping', type=int, default=10000,
        help='Early stopping')
    #
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    # linkx
    parser.add_argument('--inner_activation', action='store_true',
                        help='Whether linkV3 uses inner activation')
    parser.add_argument('--inner_dropout', action='store_true',
                        help='Whether linkV3 uses inner dropout')
    parser.add_argument('--link_init_layers_A', type=int, default=1)
    parser.add_argument('--link_init_layers_X', type=int, default=1)
    # appnp
    parser.add_argument('--gpr_alpha', type=float, default=.1,
                        help='alpha for gprgnn')
    # h2gcn
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='number of mlp layers in h2gcn')
    # used for mlpnorm
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='Weight for frobenius norm on Z.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for frobenius norm on Z-A')
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='Weight for MLP results kept')
    parser.add_argument('--delta', type=float, default=0.0,
                        help='Weight for node features, thus 1-delta for adj')
    parser.add_argument('--norm_func_id', type=int, default=2,
                        help='Function of norm layer, ids \in [1, 2]')
    parser.add_argument('--norm_layers', type=int, default=1,
                        help='Number of groupnorm layers')
    parser.add_argument('--orders_func_id', type=int, default=2,
                        help='Sum function of adj orders in norm layer, ids \in [1, 2, 3]')
    parser.add_argument('--orders', type=int, default=1,
                        help='Number of adj orders in norm layer')
    # acm
    parser.add_argument('--variant', type=int, default=1, help='Indicate ACM, GCNII variant models.')
    parser.add_argument('--structure_info', type=int, default=0, help='1 for using structure information in acmgcn+, 0 for not')
    parser.add_argument('--acm_method', type=str, 
        help='name of model (gcn, sgc, graphsage, snowball, gcnII, acmgcn, acmgcnp, acmgcnpp, acmsgc, acmgraphsage, acmsnowball, mlp)', 
        default = 'acmgcnp')
