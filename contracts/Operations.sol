// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.5.17;
pragma experimental ABIEncoderV2;

contract Operations {
    uint256 gamma_times_rho_ij = 100;
    uint256[10] public d_i_times_s_i;
    uint256[10] public gamma_times_f_i;
    uint256[10] public deposits;

    function Submit(
        uint256 _d_i_times_s_i,
        uint256 _gamma_times_f_i,
        uint256 l
    ) public payable returns (bool) {
        d_i_times_s_i[l] = _d_i_times_s_i;
        gamma_times_f_i[l] = _gamma_times_f_i;
        deposits[l] = msg.value; 
        return true;
    }

    function reward_calculate(uint256 i, uint256 client_total_num) public view returns (uint256) {
        uint256 reward;

        reward = deposits[i];
        for (uint256 j = 0; j < client_total_num; j++) {
            reward =
                reward +
                d_i_times_s_i[i] +
                gamma_times_f_i[i] -
                d_i_times_s_i[j] -
                gamma_times_f_i[j];
        }
        return reward; 
    }

    function reward_transfer(uint256 reward) public payable {
        address payable receiver_addr = payable(msg.sender);
        receiver_addr.transfer(reward);
    }
}
